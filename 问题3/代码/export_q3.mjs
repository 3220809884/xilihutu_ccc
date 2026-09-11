// Populate the original result3 template with verified solver output.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const outputDir = path.join(root, '问题3/结果');
const previewDir = process.env.Q3_PREVIEW_DIR;
if (!previewDir) throw new Error('Set Q3_PREVIEW_DIR to a writable temporary directory');
await fs.mkdir(previewDir, { recursive: true });
const payload = JSON.parse(await fs.readFile(path.join(outputDir, 'problem3_excel_data.json'), 'utf8'));
if (payload.dates.length !== 334 || payload.plan_rows.some(row => row.length !== 146) || payload.adjusted_rows.some(row => row.length !== 146)) {
  throw new Error('Invalid 334 x 144 result payload');
}
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, '题目/附件/附件5/result3.xlsx')));
const plan = workbook.worksheets.getItem('计划购电量');
const adjusted = workbook.worksheets.getItem('调整购电量');
const storage = workbook.worksheets.getItem('充放电量');
const emergency = workbook.worksheets.getItem('紧急购电量');
const serial = date => (Date.parse(`${date}T00:00:00Z`) - Date.UTC(1899, 11, 30)) / 86400000;
const close = (a, b) => Math.abs(Number(a) - Number(b)) < 1e-6;
for (const sheet of [plan, adjusted]) {
  const dates = sheet.getRange('A2:A335').values.flat();
  if (dates.some((value, index) => !close(value, serial(payload.dates[index])))) throw new Error(`${sheet.name} date mapping differs from payload`);
}
plan.getRange('B2:EQ335').values = payload.plan_rows;
adjusted.getRange('B2:EQ335').values = payload.adjusted_rows;
plan.getRange('B2:EQ335').setNumberFormat('0.0000');
adjusted.getRange('B2:EQ335').setNumberFormat('0.0000');

const storageRows = [];
for (const day of payload.storage_rows) {
  for (let block = 0; block < 6; block++) {
    storageRows.push([
      block === 0 ? serial(day.date) : null,
      `${4 * block}:00-${4 * block + 4}:00`, day.charge[block], day.discharge[block],
      block === 0 ? 0 : block === 1 ? '24:00' : null,
      block === 0 ? day.initial : block === 1 ? day.terminal : null,
    ]);
  }
}
for (let day = 1; day < 334; day++) {
  storage.getRangeByIndexes(1 + 6 * day, 0, 6, 6).copyFrom(storage.getRange('A2:F7'), 'all');
}
storage.getRangeByIndexes(1, 0, storageRows.length, 6).values = storageRows;
storage.getRange(`A2:A${storageRows.length + 1}`).setNumberFormat('yyyy/m/d');
storage.getRange(`C2:D${storageRows.length + 1}`).setNumberFormat('0.0000');
storage.getRange(`F2:F${storageRows.length + 1}`).setNumberFormat('0.0000');
storage.getRange('C:F').format.columnWidth = 17;

const eventRows = [];
for (const date of payload.dates) {
  const events = payload.events.filter(event => event.date === date);
  if (events.length === 0) eventRows.push([serial(date), '无', 0]);
  else events.forEach((event, index) => eventRows.push([index === 0 ? serial(date) : null, event.interval, event.emergency_kwh]));
}
emergency.getRange(`A2:C${Math.max(9, eventRows.length + 1)}`).clear({ applyTo: 'contents' });
for (let row = 1; row < eventRows.length; row++) {
  emergency.getRangeByIndexes(row + 1, 0, 1, 3).copyFrom(emergency.getRange('A2:C2'), 'all');
}
emergency.getRangeByIndexes(1, 0, eventRows.length, 3).values = eventRows;
emergency.getRange(`A2:A${eventRows.length + 1}`).setNumberFormat('yyyy/m/d');
emergency.getRange(`C2:C${eventRows.length + 1}`).setNumberFormat('0.0000');
emergency.getRange('A:A').format.columnWidth = 16;
emergency.getRange('B:B').format.columnWidth = 22;
emergency.getRange('C:C').format.columnWidth = 18;

for (const [sheet, rows] of [[plan, payload.plan_rows], [adjusted, payload.adjusted_rows]]) {
  const totals = sheet.getRange('EP2:EQ335').values;
  for (let index = 0; index < 334; index++) {
    if (!close(totals[index][0], rows[index][144]) || !close(totals[index][1], rows[index][145])) {
      throw new Error(`${sheet.name} total mismatch on ${payload.dates[index]}`);
    }
  }
}
const errorScan = await workbook.inspect({
  kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 100 }, summary: 'final formula error scan',
});
console.log(errorScan.ndjson);
for (const [sheetName, range, label] of [
  ['计划购电量', 'A1:F5', 'plan'], ['计划购电量', 'EL1:EQ5', 'plan_totals'],
  ['调整购电量', 'A1:F5', 'adjusted'], ['调整购电量', 'EL1:EQ5', 'adjusted_totals'],
  ['充放电量', 'A1:F13', 'storage'], ['充放电量', 'A1999:F2005', 'storage_last'],
  ['紧急购电量', 'A1:C15', 'emergency'],
]) {
  const image = await workbook.render({ sheetName, range, scale: 1.5 });
  await fs.writeFile(path.join(previewDir, `q3_${label}.png`), new Uint8Array(await image.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, 'result3.xlsx'));
console.log(JSON.stringify({ file: 'result3.xlsx', days: 334, slots: 48096, storageRows: storageRows.length, emergencyRows: eventRows.length }));
