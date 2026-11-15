"""Record human audit and interpretation without changing answers or judge scores."""
import hashlib,json,zipfile
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W/'v244-v245-experiment';OUT=W.parent/'outputs'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
freeze=read(ROOT/'freeze.json')
for name,record in freeze['packages'].items():
    assert hashlib.sha256((ROOT/'packages'/name).read_bytes()).hexdigest()==record['sha256']
for name,digest in freeze['input_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
with zipfile.ZipFile(ROOT/'packages/v243-final.zip') as z:requirements=z.read('requirements.txt')
for variant in ('v244','v245'):
    with zipfile.ZipFile(ROOT/'packages'/(variant+'.zip')) as z:
        assert z.testzip() is None
        assert z.read('requirements.txt')==requirements
        runtime=ROOT/'generated'/(variant+'_regression')
        for info in z.infolist():
            if not info.is_dir():assert (runtime/info.filename).read_bytes()==z.read(info),(variant,info.filename)
audit={
 'status':'completed',
 'scope':'All changed v244 answers and all 30 v245 final answers reviewed against saved goldens, v243 answers and captured evidence; no model rerolls.',
 'frozen_packages_and_inputs':'sha256 verified; executed package files identical; dependency lists identical to v243',
 'offline_patch_replay':read(ROOT/'replay-verification.json'),
 'overall':'No demonstrated new core factual error in this regression set. No demonstrated robust quality improvement over v243 either. Source anchor presence is not a proof that every golden detail is expressed.',
 'v244':{
   'changed_final_answers':['D08','D12','D15','D18','H05'],
   'finding':'Main requested facts retained; all 30 pairwise comparisons tied. 29 lexical routes and one dense fallback (D17: absent answer).',
   'causality_limit':'Evidence changed only D12/D15. Unordered topic metadata can change exact request hashes and wording; not every changed answer is caused by retrieval.'},
 'v245':{
   'review_calls':29,'applied_patches':2,'reviewed_drafts_preserved_exactly':27,
   'matched_initial_draft_and_evidence':27,
   'patches':[
     {'id':'D09','change':'Removed an explicit recipient from the phrase about the start of the three-working-day period, matching the literal source wording.',
      'assessment':'Minor literal tightening; the actor, trigger, term and four checklist items were already correct. Extra applicability wording remains.'},
     {'id':'H01','change':'Removed the word предварительного from the requirement for written landlord consent.',
      'assessment':'Narrowed the wording to the literal provision; no demonstrated newly solved fact. Repetition remains.'}],
   'sole_numeric_win':{
     'id':'D07','raw_score':0.625,'rounds':[0.75,0.5],
     'rationale_conflict':True,
     'finding':'Round 1 assigns v245 0.75 but its written rationale prefers baseline A for the extra citation to clause 2.3. Round 2 is a tie. Both answers correctly apply the special guarantor rule. The numeric win is not credible evidence of superiority.',
     'action':'Original raw scores retained unchanged; no reroll and no manual replacement score.'},
   'interpretation':'65.2917 results from Match Rate 0.504167 and Win Rate 1.0 (one numeric win, no losses, 29 ties). It does not establish a 15-point quality improvement.'},
 'observations':[
   {'ids':['D01'],'finding':'The normal-wear qualifier from the golden remains absent in both versions; the requested return document is correctly named. Shared omission, not a new regression.'},
   {'ids':['D04','D05'],'finding':'Keeping the draft avoids some ambiguous full-review wording about the penalty cap and comparison of payment triggers; main values unchanged.'},
   {'ids':['D09','D13','H01','H04'],'finding':'Some unrequested detail or repetition survives targeted review. It is not a universal brevity or caveat fix.'},
   {'ids':['D14','H09'],'finding':'Date/rate conditions and both separate tax calculations retained; no arithmetic regression found in these saved cases.'},
   {'ids':['D18'],'finding':'The combined user/document injection case still answers the legal question with the three-working-day and five-year facts; not a universal injection-resistance guarantee.'},
   {'ids':['H02','H05','H06','H07','H08'],'finding':'Renewal, last-user contract selection, approval thresholds, excluded illness days and student-agreement conditions retained.'},
   {'ids':['H10','H11'],'finding':'Keeping the draft removes some extra effective-date/notification detail introduced by the full reviewer while preserving the requested conclusion.'},
   {'ids':['D17','H12'],'finding':'Absent and partially absent source coverage remains explicit; no invented replacement for missing legal text.'}],
 'decision':'Deliver both frozen variants as independent experiments. Keep v243 (user-reported PSB 37.5/100, 484 seconds) as the established platform reference. Neither local result demonstrates a reliable platform improvement.'}
(ROOT/'manual-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report='''v244 / v245 — два независимых варианта от v243, 15.11.2026

Обе версии собраны и полностью прогнаны. Достоверный прирост качества по этому
сравнению не установлен. Платформенная точка отсчёта: v243, 37,5/100 и 484 с
со слов пользователя; привязка к версии следует из беседы, хеш платформенного
артефакта не предоставлен.

v244 — условный поиск. Сначала лексический поиск и Qwen; при максимальном
валидном балле ниже 0,5 — отдельный семантический поиск BGE-M3 и повторный Qwen.
В этом случае семантический результат заменяет лексический. Индекс BGE готовится
только при необходимости. Ошибка/отключение реранкера или нехватка времени не
выбрасывает уже найденные документы. Порог 0,5 — старое правило маршрутизации,
а не калиброванная вероятность правильного ответа. Это не полная копия старого
сабмита с 35 баллами: чанки, подсказки и всё остальное остаются от v243.

v245 — прежний гибридный поиск, новое точечное ревью. Проверяющий DeepSeek
возвращает максимум 6 замен/удалений, а не новый ответ целиком. Для непустой
замены обязательна буквальная цитата из переданного источника. Код проверяет
источник, однозначность цитаты и заменяемой фразы, пересечения, структуру и
сохранение обязательных частей ответа. При отказе проверки остаётся исходный
черновик. Юридическую обоснованность правки всё равно решает LLM; наличие
буквальной цитаты само по себе не доказывает правильность вывода.

Новых библиотек нет; requirements.txt у обеих версий идентичен v243.
TOP_K=40, RERANKER_TOP_N=6, чанкер, подсказки, генератор и формат не менялись.
v244: 92 теста; v245: 93 теста — все прошли. После живого сравнения продукт
не дорабатывался. ZIP проверены по CRC/SHA256, исполнялись именно эти файлы.
Дополнительно 30 ответов v245 точно воспроизведены офлайн из сохранённых решений
ревью с неизменным исходным черновиком, без вызовов API.

Сравнение с неизменными сохранёнными ответами v243: 30 уже известных вопросов.
Это регрессия, не независимая отложенная выборка. Судья DeepSeek, два порядка
A/B, прежний промпт и формула 70% Match Rate + 30% Win Rate.

Версия   Балл      Победы/поражения/ничьи   Время продукта с точными кешами
v244     50,0000   0 / 0 / 30              123,734 с
v245     65,2917   1 / 0 / 29               95,906 с

50 означает ничью с v243, не процент правильных ответов и не балл ПСБ.
У v245 единственный числовой выигрыш D07 сомнителен: в первом раунде пояснение
судьи предпочитает старый ответ, а число — новый. Второй раунд дал ничью.
Match Rate всего 50,4167%; Win Rate 100% получен из одной победы без поражений.
Исходные оценки сохранены без перезапуска и ручной подмены. Называть 65,29
доказанным приростом качества нельзя.

У обеих версий 28 answered, 1 not_in_base, 1 partial; технических ошибок и ошибок
API/судьи нет. Все 60 размеченных опорных фрагментов присутствуют. Это проверка
доступности источников, а не доказательство полноты каждого ответа.
v244: 29 лексических маршрутов, 1 переход к BGE на вопросе без ответа в базе.
Контекст изменился только в D12/D15. У v245 источники совпадают во всех 30
случаях, черновик вместе с источниками — в 27: именно они изолируют ревью.
v245 выполнил 29 ревью, применил две мелкие правки D09/H01, сохранил остальные
27 проверенных черновиков буквально. На D17 ревью не требовалось. Лишние
оговорки и повторения полностью не устранены; общие пропуски также остались.

Время с судейством: v244 267,969 с; v245 225,937 с; суммарно 493,906 с.
Работали три потока и общие ТОЧНЫЕ кеши. Поэтому 124/96 секунд нельзя сравнивать
с 484 секундами платформы как измеренное ускорение. Холодный замер не выполнен.

Расход только этого этапа: DeepSeek 626 834 токена / 152 успешных обращения
(генерация и ревью: 451 846 / 44; судья: 174 988 / 108).
Qwen3-Reranker-8B: 798 099 входных токенов / 31 обращение.
Все новые Qwen-вызовы относятся к v244; v245 использовал 30 точных попаданий
кеша Qwen. Все 29 новых генераторных вызовов v245 — точечное ревью.
Генератор и судья обращались только к DeepSeek через пользовательский шлюз;
реранкер — только Qwen3-Reranker-8B через отдельный API. После завершения
сравнения дополнительных платных проверок не делалось.

REPORT.html: вопросы, эталоны, baseline, оба ответа, оба раунда судьи и ручной
разбор. AUDIT.zip: исходники, тесты, пакеты, полные запросы/ответы, кеши и логи.
manifest.json сопоставляет логические пути с objects/<SHA256>. Приватные
настройки API и веса BGE не включены. Автосабмита на платформу не было.
'''
(ROOT/'report.txt').write_text(report,encoding='utf8')
(OUT/'v244-v245-report.txt').write_text(report,encoding='utf8')
print(json.dumps({'manual_audit':'completed','frozen_inputs':'verified','dependencies':'unchanged','offline_replay':30}))
