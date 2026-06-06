const fs = require("fs");
const { chromium } = require("playwright");

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const report = {};

  await page.goto("http://127.0.0.1:8502", { waitUntil: "domcontentloaded" });
  await page.getByRole("heading", { name: "EmoNet v7" }).waitFor({ timeout: 30000 });
  await page.getByText("Thoughts").waitFor({ timeout: 30000 });
  report.initialText = await page.locator("body").innerText();

  const input = page.getByPlaceholder("메시지를 입력하세요");
  await input.fill("지금 테스트 중이야. 너무 즉시 대답하지 않아도 돼.");
  await input.press("Enter");

  await page.waitForTimeout(1000);
  const afterOneSecond = await page.locator("body").innerText();
  const chatCountAfterOneSecond = await page.locator('[data-testid="stChatMessage"]').count();
  report.afterOneSecond = {
    hasUserText: afterOneSecond.includes("지금 테스트 중이야"),
    hasAbsorbedWithoutReply: afterOneSecond.includes("absorbed without reply"),
    hasInternalOnlyStimulation: afterOneSecond.includes("internal-only stimulation"),
    hasPendingSpeech: afterOneSecond.includes("아직 말이 올라오지 않음"),
    hasThoughts: afterOneSecond.includes("Thoughts"),
    hasEmotionModelCount: afterOneSecond.includes("Models") && afterOneSecond.includes("4"),
    chatMessageCount: chatCountAfterOneSecond,
  };

  await page.waitForTimeout(4500);
  const afterDelay = await page.locator("body").innerText();
  const chatCountAfterDelay = await page.locator('[data-testid="stChatMessage"]').count();
  report.afterDelay = {
    hasUserText: afterDelay.includes("지금 테스트 중이야"),
    hasAbsorbedWithoutReply: afterDelay.includes("absorbed without reply"),
    hasInternalOnlyStimulation: afterDelay.includes("internal-only stimulation"),
    hasDelayedSpeechTrace: afterDelay.includes("event=delayed_speech"),
    hasThoughts: afterDelay.includes("Thoughts"),
    hasThoughtDialogue: /감각|경계|정리|기억/.test(afterDelay),
    chatMessageCount: chatCountAfterDelay,
  };
  report.finalText = afterDelay;

  await page.screenshot({ path: "v7/outputs/streamlit_gui_live_check.png", fullPage: true });
  fs.writeFileSync("v7/outputs/streamlit_gui_live_check.json", JSON.stringify(report, null, 2), "utf8");
  await browser.close();

  if (report.afterOneSecond.hasAbsorbedWithoutReply || report.afterDelay.hasAbsorbedWithoutReply) {
    throw new Error("internal absorbed-without-reply text leaked into the GUI");
  }
  if (report.afterOneSecond.hasInternalOnlyStimulation || report.afterDelay.hasInternalOnlyStimulation) {
    throw new Error("internal-only stimulation text leaked into the GUI");
  }
  if (!report.afterDelay.hasUserText) {
    throw new Error("user text was not visible after the delayed speech window");
  }
  if (report.afterDelay.chatMessageCount < 2) {
    throw new Error("delayed speech did not become a visible assistant chat message");
  }
  if (!report.afterDelay.hasThoughts || !report.afterDelay.hasThoughtDialogue) {
    throw new Error("thought dialogue was not visible after interaction");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
