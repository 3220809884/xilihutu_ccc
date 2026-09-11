// Fill the original template with verified solver output using the bundled runtime.
// NODE_PATH must point to the bundled node_modules directory (see README.md).
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const outputDir = path.join(root, '问题2/结果');
const previewDir = process.env.Q2_PREVIEW_DIR;
if (!previewDir) throw new Error('Set Q2_PREVIEW_DIR to a writable temporary preview directory');
await fs.mkdir(previewDir, { recursive: true });
const payload = JSON.parse(await fs.readFile(path.join(outputDir, 'problem2_excel_data.json'), 'utf8'));
if (payload.dates.length !== 334 || payload.plan_rows.some(r => r.length !== 146)) throw new Error('Invalid 334 x 144 data grid');
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, '题目/附件/附件5/result2.xlsx')));
const pricesWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, '题目/附件/附件1.xlsx')));
const prices = pricesWorkbook.worksheets.getItemAt(0).getRange('B2:B145').values.flat();
const plan = workbook.worksheets.getItem('计划购电量');
const storage = workbook.worksheets.getItem('充放电量');
const emergency = workbook.worksheets.getItem('紧急购电量');
const serial = date => (Date.parse(date+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000;
const close = (a,b) => Math.abs(a-b) < 1e-6;
// Keep original date cells and all 144 source labels, including the documented offset.
const templateDates = plan.getRange('A2:A335').values.flat();
if (templateDates.some((d,i) => !close(Number(d), serial(payload.dates[i])))) throw new Error('Template date mapping differs');
plan.getRange('B2:EQ335').values = payload.plan_rows;
plan.getRange('B2:EQ335').setNumberFormat('0.0000');
plan.getRange('EP2:EP335').formulas = payload.dates.map((_,i) => [`=SUM(B${i+2}:EO${i+2})`]);
const priceArray = '{'+prices.join(',')+'}';
plan.getRange('EQ2:EQ335').formulas = payload.dates.map((_,i) => [`=SUMPRODUCT(B${i+2}:EO${i+2},${priceArray})`]);
// The template has two example days and an ellipsis: extend its six-row block.
const storageRows = [];
for (const day of payload.storage_rows) {
  for (let b=0;b<6;b++) storageRows.push([
    b===0 ? serial(day.date) : null, `${4*b}:00-${4*b+4}:00`, day.charge[b], day.discharge[b],
    b===0 ? 0 : b===1 ? '24:00' : null, b===0 ? day.initial : b===1 ? day.terminal : null,
  ]);
}
// Copy formats before clearing the template's illustrative values.
for (let i=1;i<334;i++) storage.getRangeByIndexes(1+6*i,0,6,6).copyFrom(storage.getRange('A2:F7'),'all');
storage.getRangeByIndexes(1,0,storageRows.length,6).values = storageRows;
storage.getRange(`A2:A${storageRows.length+1}`).setNumberFormat('yyyy/m/d');
storage.getRange(`C2:D${storageRows.length+1}`).setNumberFormat('0.0000');
storage.getRange(`F2:F${storageRows.length+1}`).setNumberFormat('0.0000');
storage.getRange('C:F').format.columnWidth = 17;
const eventRows = [];
for (const date of payload.dates) {
  const events = payload.events.filter(e => e.date === date);
  if (events.length === 0) eventRows.push([serial(date), '无', 0]);
  else events.forEach((e,i) => eventRows.push([i===0 ? serial(date) : null,e.interval,e.emergency_kwh]));
}
emergency.getRange(`A2:C${Math.max(11,eventRows.length+1)}`).clear({applyTo:'contents'});
for (let i=1;i<eventRows.length;i++) emergency.getRangeByIndexes(i+1,0,1,3).copyFrom(emergency.getRange('A2:C2'),'all');
emergency.getRangeByIndexes(1,0,eventRows.length,3).values = eventRows;
emergency.getRange(`A2:A${eventRows.length+1}`).setNumberFormat('yyyy/m/d');
emergency.getRange(`C2:C${eventRows.length+1}`).setNumberFormat('0.0000');
emergency.getRange('A:A').format.columnWidth=16;
emergency.getRange('B:B').format.columnWidth=22;
emergency.getRange('C:C').format.columnWidth=18;
// Totals are formula-based, checked against independent Python results before export.
const totals = plan.getRange('EP2:EQ335').values;
for (let i=0;i<334;i++) {
  if (!close(totals[i][0],payload.plan_rows[i][144]) || !close(totals[i][1],payload.plan_rows[i][145])) throw new Error(`Total mismatch on ${payload.dates[i]}`);
}
console.log((await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'Formula error scan'})).ndjson);
for (const [sheet,range,label] of [
  ['计划购电量','A1:F5','plan'],['计划购电量','EL1:EQ5','plan_totals'],
  ['充放电量','A1:F13','storage'],['紧急购电量','A1:C15','emergency'],
  ['充放电量','A1999:F2005','storage_last_day'],
]) {
  const img=await workbook.render({sheetName:sheet,range,scale:1.5});
  await fs.writeFile(path.join(previewDir,`q2_${label}.png`),new Uint8Array(await img.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir,'result2.xlsx'));
console.log(JSON.stringify({file:'result2.xlsx',planned_days:334,planned_slots:48096,storage_rows:storageRows.length,emergency_rows:eventRows.length}));
