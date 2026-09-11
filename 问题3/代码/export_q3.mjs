// Populate a copy of the official four-sheet result3 template.
// NODE_PATH must point to the Codex bundled node_modules directory.
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(require.resolve("@oai/artifact-tool")).href);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const out = path.join(root, "问题3/结果");
const previews = process.env.Q3_PREVIEW_DIR;
if (!previews) throw new Error("Set Q3_PREVIEW_DIR to a writable temporary directory");
await fs.mkdir(previews, { recursive: true });
const data = JSON.parse(await fs.readFile(path.join(out, "problem3_excel_data.json"), "utf8"));
if (data.dates.length !== 334 || data.intervals.length !== 144) throw new Error("Expected 334 days and 144 slots");
const w = await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root, "题目/附件/附件5/result3.xlsx")));
const serial = date => (Date.parse(`${date}T00:00:00Z`) - Date.UTC(1899, 11, 30)) / 86400000;
const close = (a, b) => Math.abs(Number(a) - Number(b)) < 1e-4;
const prices = `{${data.prices.join(",")}}`;
for (const [name, values] of [["计划购电量", data.plan_rows], ["调整购电量", data.adjusted_rows]]) {
  const sheet = w.worksheets.getItem(name);
  if (sheet.getRange("A2:A335").values.some((r, i) => !close(r[0], serial(data.dates[i])))) throw new Error("Template date mismatch");
  // Correct the COPY's known +10-minute header defect. Original is untouched.
  sheet.getRange("B1:EO1").values = [data.intervals];
  sheet.getRange("B2:EQ335").values = values;
  sheet.getRange("B2:EQ335").setNumberFormat("0.0000");
  sheet.getRange("EP2:EP335").formulas = data.dates.map((_, i) => [`=SUM(B${i+2}:EO${i+2})`]);
  sheet.getRange("EQ2:EQ335").formulas = data.dates.map((_, i) => {
    const r = i + 2;
    if (name === "计划购电量") return [`=SUMPRODUCT(B${r}:EO${r},${prices})`];
    // Under this settlement rule, downwards revisions are dominated. Verify
    // that premise for every exported cell before using the simplified fee.
    if (data.adjusted_rows[i].slice(0,144).some((q,t)=>q < data.plan_rows[i][t]-1e-7)) throw new Error("Downward revision requires general piecewise fee export");
    return [`=1.5*(SUMPRODUCT(B${r}:EO${r},${prices})-'计划购电量'!EQ${r})`];
  });
  if (name === "调整购电量") sheet.getRange("EQ1").values = [["全天调整相关费用（元）"]];
  sheet.getRange("B1:EO1").format.columnWidth = 16;
  sheet.getRange("EP1:EQ335").format.columnWidth = 23;
  sheet.getRange("A1:EQ1").format.rowHeight = 32;
  sheet.freezePanes.freezeRows(1);
}
const storage = w.worksheets.getItem("充放电量");
const storageRows = [];
for (const day of data.storage_rows) {
  for (let b = 0; b < 6; b++) storageRows.push([b === 0 ? serial(day.date) : null, `${b*4}:00-${b*4+4}:00`, day.charge[b], day.discharge[b], b === 0 ? "0:00" : b === 1 ? "24:00" : null, b === 0 ? day.initial : b === 1 ? day.terminal : null]);
}
for (let i=1; i<334; i++) storage.getRangeByIndexes(1+6*i,0,6,6).copyFrom(storage.getRange("A2:F7"),"all");
storage.getRangeByIndexes(1,0,storageRows.length,6).values = storageRows;
storage.getRange("A2:A2005").setNumberFormat("yyyy/m/d");
storage.getRange("C2:D2005").setNumberFormat("0.0000");
storage.getRange("F2:F2005").setNumberFormat("0.0000");
storage.getRange("A1:F2005").format.verticalAlignment = "center";
storage.getRange("C1:F2005").format.columnWidth = 17;
storage.getRange("A1:B2005").format.horizontalAlignment = "center";
storage.getRange("C2:D2005").format.horizontalAlignment = "right";
storage.getRange("F2:F2005").format.horizontalAlignment = "right";
storage.getRange("A1:F2005").format.borders = {insideVertical:{style:"thin",color:"#000000"},left:{style:"thin",color:"#000000"},right:{style:"thin",color:"#000000"}};
for(let i=0; i<334; i++) storage.getRangeByIndexes(1+6*i,0,6,6).format.borders = {preset:"outside",style:"thin",color:"#000000"};
storage.freezePanes.freezeRows(1);
const emergency = w.worksheets.getItem("紧急购电量");
const grouped = new Map(data.dates.map(d=>[d,[]]));
for(const event of data.events) grouped.get(event.date).push(event);
const eventRows = [];
for(const date of data.dates){
  const list=grouped.get(date);
  if(!list.length) eventRows.push([serial(date),"无",0]);
  else list.forEach((event,i)=>eventRows.push([i===0?serial(date):null,event.interval,event.emergency_kwh]));
}
emergency.getRange(`A2:C${Math.max(11,eventRows.length+1)}`).clear({applyTo:"contents"});
for(let i=1;i<eventRows.length;i++) emergency.getRangeByIndexes(i+1,0,1,3).copyFrom(emergency.getRange("A2:C2"),"all");
emergency.getRangeByIndexes(1,0,eventRows.length,3).values=eventRows;
emergency.getRange(`A2:A${eventRows.length+1}`).setNumberFormat("yyyy/m/d");
emergency.getRange(`C2:C${eventRows.length+1}`).setNumberFormat("0.0000");
emergency.getRange(`A1:A${eventRows.length+1}`).format.columnWidth=16;
emergency.getRange(`B1:B${eventRows.length+1}`).format.columnWidth=23;
emergency.getRange(`C1:C${eventRows.length+1}`).format.columnWidth=18;
emergency.getRange(`A1:C${eventRows.length+1}`).format.verticalAlignment="center";
emergency.getRange(`A1:B${eventRows.length+1}`).format.horizontalAlignment="center";
emergency.getRange(`C2:C${eventRows.length+1}`).format.horizontalAlignment="right";
emergency.getRange(`A1:C${eventRows.length+1}`).format.borders={preset:"all",style:"thin",color:"#000000"};
emergency.freezePanes.freezeRows(1);
w.recalculate();
for(const [name, expected] of [["计划购电量",data.plan_rows],["调整购电量",data.adjusted_rows]]){
  const actual=w.worksheets.getItem(name).getRange("EP2:EQ335").values;
  actual.forEach((r,i)=>{if(!close(r[0],expected[i][144])||!close(r[1],expected[i][145]))throw new Error(`${name} daily totals mismatch ${data.dates[i]}: ${JSON.stringify(r)}`);});
}
// A real input-change check of the fee dependency, then restore before export.
const adjustmentSheet=w.worksheets.getItem("调整购电量");
const original=adjustmentSheet.getRange("B2").values[0][0];
const originalFee=adjustmentSheet.getRange("EQ2").values[0][0];
adjustmentSheet.getRange("B2").values=[[original+10]];
w.recalculate();
if(!close(adjustmentSheet.getRange("EQ2").values[0][0],originalFee+15*data.prices[0]))throw new Error("Adjustment fee recalculation failed");
adjustmentSheet.getRange("B2").values=[[original]];
w.recalculate();
if(!close(adjustmentSheet.getRange("EQ2").values[0][0],originalFee))throw new Error("Adjustment fee restoration failed");
console.log((await w.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:20},summary:"Formula error scan"})).ndjson);
for(const [sheet,range,name] of [["计划购电量","A1:F5","plan"],["计划购电量","EN1:EQ5","plan_totals"],["调整购电量","AK1:AP5","adjusted"],["调整购电量","EN1:EQ5","adjustment_totals"],["充放电量","A1:F13","storage"],["紧急购电量","A1:C15","emergency"]]){
  const b=await w.render({sheetName:sheet,range,scale:1.5});
  await fs.writeFile(path.join(previews,`q3_${name}.png`),new Uint8Array(await b.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(w)).save(path.join(out,"result3.xlsx"));
// The runtime may emit a large inspection sidecar. Keep it with QA intermediates.
const inspection=path.join(out,"result3.xlsx.inspect.ndjson");
try { await fs.rename(inspection,path.join(previews,"result3.xlsx.inspect.ndjson")); }
catch(error) { if(error.code!=="ENOENT") throw error; }
console.log(JSON.stringify({path:path.join(out,"result3.xlsx"),days:334,storageRows:storageRows.length,eventRows:eventRows.length}));
