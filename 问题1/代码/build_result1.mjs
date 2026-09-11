import fs from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = new URL("../../", import.meta.url);
const templatePath = new URL("题目/附件/附件5/result1.xlsx", root);
const schedulePath = new URL("问题1/结果/调度数据.json", root);
const summaryPath = new URL("问题1/结果/求解摘要.json", root);
const outputPath = new URL("问题1/结果/result1.xlsx", root);
const previewDir = new URL("问题1/结果/预览/", root);

const workbook = await SpreadsheetFile.importXlsx(
  await FileBlob.load(fileURLToPath(templatePath)),
);

if (process.argv[2] === "inspect") {
  const info = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 5000,
    tableMaxRows: 8,
    tableMaxCols: 6,
  });
  console.log(info.ndjson);
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of ["计划购电量", "充放电量"]) {
    const image = await workbook.render({ sheetName, autoCrop: "all", scale: 1.5, format: "png" });
    await fs.writeFile(
      new URL(`${sheetName}-原模板.png`, previewDir),
      new Uint8Array(await image.arrayBuffer()),
    );
  }
  process.exit(0);
}

const schedule = JSON.parse(await fs.readFile(schedulePath, "utf8"));
const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
if (schedule.length !== 144) {
  throw new Error(`调度数据应有144行，实际为${schedule.length}行`);
}

const purchaseSheet = workbook.worksheets.getItem("计划购电量");
const storageSheet = workbook.worksheets.getItem("充放电量");

purchaseSheet.getRange("B2:B145").values = schedule.map((row) => [row.grid_purchase_kwh]);
purchaseSheet.getRange("B2:B145").format.numberFormat = "0.0000";

const groups = summary.four_hour_totals;
storageSheet.getRange("B2:C7").values = groups.map((row) => [row.charge_kwh, row.discharge_kwh]);
storageSheet.getRange("B2:C7").format.numberFormat = "0.0000";
storageSheet.getRange("E2:E3").values = [
  [summary.energy_initial_kwh],
  [summary.energy_final_kwh],
];
storageSheet.getRange("E2:E3").format.numberFormat = "0.0000";

workbook.recalculate();

const keyValues = await workbook.inspect({
  kind: "table",
  range: "充放电量!A1:E7",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 6,
  maxChars: 5000,
});
console.log(keyValues.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["计划购电量", "充放电量"]) {
  const image = await workbook.render({ sheetName, autoCrop: "all", scale: 1.5, format: "png" });
  await fs.writeFile(
    new URL(`${sheetName}-结果.png`, previewDir),
    new Uint8Array(await image.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(fileURLToPath(outputPath));
console.log(`saved ${outputPath.pathname}`);
