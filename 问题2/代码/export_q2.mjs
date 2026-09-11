// Fill the official result2.xlsx template with verified Problem-2 outputs.
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = await import(
  pathToFileURL(require.resolve("@oai/artifact-tool")).href
);

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const outputDir = path.join(root, "问题2/结果");
const previewDir = process.env.Q2_PREVIEW_DIR;
if (!previewDir) throw new Error("Set Q2_PREVIEW_DIR to a writable temporary directory");
await fs.mkdir(previewDir, { recursive: true });

const payload = JSON.parse(
  await fs.readFile(path.join(outputDir, "problem2_excel_data.json"), "utf8"),
);
if (payload.dates.length !== 334 || payload.plan_rows.some((row) => row.length !== 146)) {
  throw new Error("Expected 334 dates and 144 slot values plus two daily totals");
}

const workbook = await SpreadsheetFile.importXlsx(
  await FileBlob.load(path.join(root, "题目/附件/附件5/result2.xlsx")),
);
const priceBook = await SpreadsheetFile.importXlsx(
  await FileBlob.load(path.join(root, "题目/附件/附件1.xlsx")),
);
const prices = priceBook.worksheets.getItemAt(0).getRange("B2:B145").values.flat();
if (prices.length !== 144 || prices.some((value) => !Number.isFinite(Number(value)))) {
  throw new Error("Attachment-1 fixed price vector is not a valid 144-slot series");
}

const plan = workbook.worksheets.getItem("计划购电量");
const storage = workbook.worksheets.getItem("充放电量");
const emergency = workbook.worksheets.getItem("紧急购电量");
const serial = (date) =>
  (Date.parse(`${date}T00:00:00Z`) - Date.UTC(1899, 11, 30)) / 86400000;
const close = (left, right) => Math.abs(Number(left) - Number(right)) < 1e-5;

// The template labels are retained. Values are mapped by slot position:
// B is slot 0 = 00:00--00:10, even though the source header starts at 0:10--0:20.
const templateDates = plan.getRange("A2:A335").values.flat();
if (templateDates.some((value, index) => !close(value, serial(payload.dates[index])))) {
  throw new Error("Template dates do not match 2025-02-01 through 2025-12-31");
}
plan.getRange("B2:EQ335").values = payload.plan_rows;
plan.getRange("B2:EQ335").setNumberFormat("0.0000");
plan.getRange("EP2:EP335").formulas = payload.dates.map((_, index) => [
  `=SUM(B${index + 2}:EO${index + 2})`,
]);
const priceArray = `{${prices.join(",")}}`;
plan.getRange("EQ2:EQ335").formulas = payload.dates.map((_, index) => [
  `=SUMPRODUCT(B${index + 2}:EO${index + 2},${priceArray})`,
]);

const storageRows = [];
for (const day of payload.storage_rows) {
  for (let block = 0; block < 6; block += 1) {
    storageRows.push([
      block === 0 ? serial(day.date) : null,
      `${4 * block}:00-${4 * block + 4}:00`,
      day.charge[block],
      day.discharge[block],
      block === 0 ? "0:00" : block === 1 ? "24:00" : null,
      block === 0 ? day.initial : block === 1 ? day.terminal : null,
    ]);
  }
}
if (storageRows.length !== 334 * 6) throw new Error("Storage block row count mismatch");
for (let index = 1; index < 334; index += 1) {
  storage
    .getRangeByIndexes(1 + 6 * index, 0, 6, 6)
    .copyFrom(storage.getRange("A2:F7"), "all");
}
storage.getRangeByIndexes(1, 0, storageRows.length, 6).values = storageRows;
storage.getRange(`A2:A${storageRows.length + 1}`).setNumberFormat("yyyy/m/d");
storage.getRange(`C2:D${storageRows.length + 1}`).setNumberFormat("0.0000");
storage.getRange(`F2:F${storageRows.length + 1}`).setNumberFormat("0.0000");
storage.getRange(`A1:F${storageRows.length + 1}`).format.verticalAlignment = "center";
storage.getRange(`A1:B${storageRows.length + 1}`).format.horizontalAlignment = "center";
storage.getRange(`E1:E${storageRows.length + 1}`).format.horizontalAlignment = "center";
storage.getRange(`C2:D${storageRows.length + 1}`).format.horizontalAlignment = "right";
storage.getRange(`F2:F${storageRows.length + 1}`).format.horizontalAlignment = "right";
storage.getRange(`C1:F${storageRows.length + 1}`).format.columnWidth = 17;
storage.getRange(`A1:F${storageRows.length + 1}`).format.borders = {
  insideVertical: { style: "thin", color: "#000000" },
  left: { style: "thin", color: "#000000" },
  right: { style: "thin", color: "#000000" },
};
storage.getRange("A1:F1").format.borders = {
  preset: "all",
  style: "thin",
  color: "#000000",
};
for (let index = 0; index < 334; index += 1) {
  const firstRow = 2 + 6 * index;
  storage.getRange(`A${firstRow}:F${firstRow + 5}`).format.borders = {
    preset: "outside",
    style: "thin",
    color: "#000000",
  };
}

const eventsByDate = new Map(payload.dates.map((date) => [date, []]));
for (const event of payload.events) eventsByDate.get(event.date).push(event);
const eventRows = [];
for (const date of payload.dates) {
  const events = eventsByDate.get(date);
  if (events.length === 0) eventRows.push([serial(date), "无", 0]);
  else {
    events.forEach((event, index) => {
      eventRows.push([
        index === 0 ? serial(date) : null,
        event.interval,
        event.emergency_kwh,
      ]);
    });
  }
}
const emergencyEnd = Math.max(9, eventRows.length + 1);
emergency.getRange(`A2:C${emergencyEnd}`).clear({ applyTo: "contents" });
for (let index = 1; index < eventRows.length; index += 1) {
  emergency
    .getRangeByIndexes(index + 1, 0, 1, 3)
    .copyFrom(emergency.getRange("A2:C2"), "all");
}
emergency.getRangeByIndexes(1, 0, eventRows.length, 3).values = eventRows;
emergency.getRange(`A2:A${eventRows.length + 1}`).setNumberFormat("yyyy/m/d");
emergency.getRange(`C2:C${eventRows.length + 1}`).setNumberFormat("0.0000");
emergency.getRange(`A1:C${eventRows.length + 1}`).format.verticalAlignment = "center";
emergency.getRange(`A1:B${eventRows.length + 1}`).format.horizontalAlignment = "center";
emergency.getRange(`C2:C${eventRows.length + 1}`).format.horizontalAlignment = "right";
emergency.getRange(`A1:A${eventRows.length + 1}`).format.columnWidth = 16;
emergency.getRange(`B1:B${eventRows.length + 1}`).format.columnWidth = 22;
emergency.getRange(`C1:C${eventRows.length + 1}`).format.columnWidth = 18;

workbook.recalculate();
const totals = plan.getRange("EP2:EQ335").values;
for (let index = 0; index < 334; index += 1) {
  if (
    !close(totals[index][0], payload.plan_rows[index][144]) ||
    !close(totals[index][1], payload.plan_rows[index][145])
  ) {
    throw new Error(`Plan total mismatch on ${payload.dates[index]}`);
  }
}

console.log(
  (
    await workbook.inspect({
      kind: "match",
      searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
      options: { useRegex: true, maxResults: 100 },
      summary: "final formula error scan",
    })
  ).ndjson,
);
for (const [sheet, range, name] of [
  ["计划购电量", "A1:H5", "plan_start"],
  ["计划购电量", "EL1:EQ5", "plan_totals"],
  ["充放电量", "A1:F13", "storage_start"],
  ["充放电量", "A1993:F2005", "storage_end"],
  ["紧急购电量", "A1:C20", "emergency"],
]) {
  const image = await workbook.render({ sheetName: sheet, range, scale: 1.5 });
  await fs.writeFile(
    path.join(previewDir, `q2_${name}.png`),
    new Uint8Array(await image.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "result2.xlsx");
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
console.log(
  (
    await saved.inspect({
      kind: "sheet",
      include: "id,name",
      maxChars: 3000,
    })
  ).ndjson,
);
console.log(
  JSON.stringify({
    file: outputPath,
    plannedDays: 334,
    plannedSlots: 334 * 144,
    storageRows: storageRows.length,
    emergencyRows: eventRows.length,
  }),
);
