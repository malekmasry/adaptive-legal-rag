"use strict";
const find = (id) => document.getElementById(id);
const question = find("question");
const submit = find("submit");
const copy = find("copy");
const labels = {
  answered: "Ответ по документам",
  partial: "Ответ по доступным сведениям",
  not_in_base: "В базе недостаточно сведений",
  clarification: "Нужно уточнение",
  out_of_domain: "Вопрос вне базы",
  technical_error: "Сервис временно недоступен",
};

function count() {
  find("counter").textContent = question.value.length + " / 4000";
}

function show(title, text, canCopy) {
  find("idle").hidden = true;
  find("result").hidden = false;
  find("status").textContent = title;
  find("answer").textContent = text;
  copy.textContent = "Скопировать";
  copy.hidden = !canCopy;
}

async function examples() {
  const response = await fetch("/api/examples");
  if (!response.ok) throw new Error("Примеры недоступны.");
  const data = await response.json();
  find("mode").textContent = data.demo ? "Офлайн-демо" : "Ответы по базе";
  for (const item of data.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.label + " →";
    button.addEventListener("click", () => {
      question.value = item.question;
      count();
      question.focus();
    });
    find("examples").append(button);
  }
}

question.addEventListener("input", count);
find("ask").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled || !question.value.trim()) return;
  submit.disabled = true;
  show("Ищем основание в документах", "Подождите, пока ассистент подготовит ответ.", false);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: question.value.trim()}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Сервис временно недоступен.");
    show(labels[result.status] || "Ответ", result.answer, result.status !== "technical_error");
  } catch (error) {
    show("Запрос не выполнен", error.message, false);
  } finally {
    submit.disabled = false;
  }
});

copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(find("answer").textContent);
    copy.textContent = "Скопировано";
  } catch {
    copy.textContent = "Выделите текст для копирования";
  }
});
examples().catch(() => { find("examples").textContent = "Примеры временно недоступны."; });
