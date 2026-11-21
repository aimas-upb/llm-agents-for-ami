# Pasul de structurare: ce face, cum functioneaza M3, trie (G) si GENRE (H), si ce arata rezultatele de pe fep

Data: 2026-11-17. Autor: Coder. Surse: codul din repo (commit `22ad219c`), inregistrarile celor 4 rulari ECAI-400 de pe fep
(`tests/homebench/results/harness/`: `fep400.M3`, `fep.trie-v1`, `fep.genre-v1`, `fep.trie-v2`), scriptul
`work/reports/structuring_analysis.py` (scris pentru acest raport, doar stdlib, fara apel LLM) si dosarul LLM-for-AmI
(`decisions.md` D55-D61, `findings.md`, `progress.md`).

Acoperire (verificata 2026-11-21): cifrele de mai jos (M3, G, H, G2 pe ECAI-400) coincid cu cele regenerate acum de `structuring_analysis.py` din `stats/` (regenerat pentru toate rularile, rezultate identice).
Rularile ulterioare (H2, G2-M1, H2-M1 cu Qwen3-8B la structurare) nu sunt in acest document; ele sunt in `experimente_complet.md` (sectiunile 8, 12, 13), care contine si timpul si costul tuturor rularilor.

Conventii: "slot" = un camp al JSON-ului produs de structurare. "IRI" / "CURIE" = numele unui lucru din ontologie scris cu
prefix (`homeont:Foyer`, `saref:SetAbsoluteLevelCommand`). "Episod" = o cerere din ECAI-400. "Parte" = o
comanda din gold-ul unui episod (un episod `hb3` are 3 parti). Cifrele din tabele sunt calculate de
`structuring_analysis.py` din `errors_episodes.csv`, `errors_parts.csv` si din inregistrarile episoadelor; episoadele scorate ca
"infra" (API/simulator picat) sunt scoase din toate ratele.

---

## 0. Pe scurt (de citit daca ai doar 2 minute)

1. **Pasul de structurare** primeste o cerere atomica ("Set the brightness of the light to 10 in the master bedroom"), eticheta
   ei de la segmentare (`explicit, achievement`) si o vedere a casei (camere, dispozitive, actiuni, cu numele lor exacte).
   Produce un JSON care **ancoreaza cererea in vocabularul casei**: ce camera (`homeont:MasterBedroom`), ce dispozitiv
   (`master_bedroom_light`), ce actiune (`brightness`, clasa `saref:SetAbsoluteLevelCommand`), ce valoare (`10`). Planificatorul
   compara aceste siruri cu casa: un nume scris cu o litera sau un prefix gresit nu gaseste nimic.
2. Pasul are **doua sarcini care se trag in directii opuse**: (a) pe cererile fezabile, sa scrie *exact* numele care exista;
   (b) pe cererile infezabile (dispozitivul sau atributul nu exista), sa **spuna ca nu exista**, nu sa gaseasca ceva apropiat.
3. **M3 (varianta implicita)**: un singur apel liber la DeepSeek-R1-Distill-Qwen-7B. Modelul gandeste, apoi scrie JSON.
   Intelege cererea, dar scrie numele "ca oamenii" (`Foyer`, `DimmableLight`, `Master bedroom`), iar planificatorul nu accepta nimic
   din asta: pe fezabile trece 12,4% din parti. Pe infezabile "trece" 95,7% dar **din greseala**: refuza aproape totul, pentru
   ca numele stricate nu gasesc dispozitivul.
4. **Trie (G)** si **GENRE (H)** sunt aceeasi idee (decodare constransa, De Cao et al. 2021): cand modelul scrie valoarea unui
   slot-nume, ii permiti doar sirurile care exista in casa. G o emuleaza peste Ollama (corectie dupa fapt, token cu token, 20
   candidati pe pas, greedy); H o face real, pe un server HuggingFace, mascand logits-urile intregului vocabular, cu beam search.
5. **Ce au castigat si ce au pierdut** (pe cele 400 de episoade, fep): M3 22,6% -> G 34,3% -> H 32,0% -> **G2 43,8%**. Pe fezabile
   cresc enorm (partile fezabile: 12,4% -> 55-64%). Pe infezabile pierd: constrangerea *inlocuieste* un dispozitiv absent cu
   cel mai apropiat dispozitiv real. G2 = G + regula evidentei (D58) + **abstinenta** (dispozitiv absent din camera => "I can't find
   the X"). H2 (acelasi lucru pentru GENRE) a rulat intre timp (20 nov, job 302815): **41,3%** pe ECAI-400, nu se distinge de G2
   (p = 0,32); vezi `experimente_complet.md` (tabelele din acest raport nu includ H2).
6. **Probleme noi gasite in acest raport** (nu erau in analizele anterioare): regula evidentei din G2 **schimba camera** pe care
   modelul a scris-o corect (36 de episoade in G2, 9 in G; 31 din cele 36 pica); continuarea libera dupa o corectie produce uneori JSON
   duplicat (10 episoade cu eroare de parsare in G2); abstinenta se declanseaza gresit pe aspirator (`Rvc`, 3 episoade).
7. **Cea mai mare pierdere ramasa nu tine de decodare**: la cererile relative ("Increase ... by 34%") modelul scrie `goal_effect:
   "set"` in loc de `"modify"`, iar acestea pica toate (G2: 0 din 20).

---

## 1. Pasul de structurare in pipeline

### 1.1 Pipeline-ul cap-coada

```
cererea utilizatorului (text)
   |
   v  1. SEGMENTARE            qwen3-8b          "cate intentii atomice sunt si de ce tip"
   |     -> intentii: {text, type, qualifiers}   type: GOAL_REQUEST | ENV_STATE_REQUEST | ENV_CAPABILITIES_REQUEST
   |                                             qualifiers: explicit|incomplete|ambiguous, achievement|maintenance,
   |                                                         *_dependency
   v  2. STRUCTURARE (un behaviour pe fiecare GOAL_REQUEST)       <-- pasul tau
   |     a. rutare (goal_routing): simple | dependency si cat context e nevoie
   |     b. contextul casei de la EnvExplorer, restrans la ce numeste cererea
   |     c. UN apel LLM  -> JSON -> GoalStructure
   |     d. rezolvare (goal_resolution): goal-urile incomplete se promoveaza la explicit
   |        daca casa admite o singura lectura; "device missing" -> intrebare
   v  3. CLARIFICARI           goal-urile fara dispozitiv/actiune/camera, sau cu dispozitiv lipsa, NU ajung la planificator;
   |                           utilizatorul primeste o intrebare ("I can't find the humidifier in the store room.")
   v  4. PLANIFICARE           planificatorul explicit (qwen2.5-coder-7b) / workflow general / modify-recipe; produce un arbore
   |                           de comportament SAU "impossible" SAU o eroare
   v  5. EXECUTIE              in simulatorul HomeBench (HBTD), stare finala
   v  6. SCORARE               comparatie cu gold-ul: fiecare parte gold trebuie indeplinita exact
```

Modelele din M3 (`homebench.ecai.M3.yaml`): segmentare Qwen3-8B, **structurare DeepSeek-R1-Distill-Qwen-7B**, planificare
Qwen2.5-Coder-7B, profiling DS-R1-7B, guarding DS-R1-1.5B. Toate la `temperature: 0`, toate pe Ollama.

Codul pasului: `ami_agents/agents/user_assistant/behaviours/goal_structuring.py` (165 de linii; `run`, `_structure`,
`_resolve`).

### 1.2 Input-ul exact

Apelul LLM are doua mesaje (`goal_structuring.py:_structure`):

| mesaj | continut |
|---|---|
| `system` | regulile structurarii (`GOAL_SPEC_RULES` sau varianta `dependency`) + un ghid al formatului contextului + **contextul casei** + forma output-ului. 19.202 de caractere in exemplul `hb1_feasible_seed_69_fixed` (depinde de context) |
| `user` | `Request: <textul intentiei>` si `Segmenter qualifiers: explicit, achievement` |

Elementele input-ului:

- **`intent_text`**: bucata din cerere care apartine acestui goal (iesirea segmentarii).
- **`qualifiers`**: etichetele segmentarii. Decid ruta (`goal_routing.route_goal`): *achievement* fara dependenta =
  `simple` (un singur goal; 98,9% din goal-urile din M3), orice altceva (dependenta logica/temporala/secventiala, maintenance,
  etichete lipsa) = `dependency` (goal-uri `G1, G2...` plus predicate `P1...` cu rol de guard/trigger/scope...).
- **contextul casei** (`pipeline.fetch_goal_context_data` -> EnvExplorer, `env_explorer/utils/actuation_context.py`): ce
  se poate *face* in casa, nu toata descrierea ei. Pentru o casa mica: tot; pentru una mare: camerele pe care le numeste cererea
  (potrivire de siruri cu etichetele ontologiei, fara LLM), plus camerele care au familia de dispozitive numita. Vine in doua
  forme (`GoalContext`): `text` (cel din prompt) si `data` (aceeasi informatie ca structura, de care au nevoie triul si
  alte variante). Un fragment real din promptul M3 (`hb1_feasible_seed_69_fixed`):

  ```
  Workspace
    name: "Master Bedroom"
    room class: homeont:MasterBedroom
    Artifact
      name: "master_bedroom_aromatherapy"
      class: homeont:Aromatherapy (class label: "Aromatherapy")
      Action
        name: "onOff"
        command class: homeont:SetOnOffCommand
        input: direct value; type: js:BooleanSchema
        acts upon: property "onOff" (class: homeont:OnOff)
  ```

Fiecare dintre aceste valori din context e un nume pe care modelul trebuie sa-l copieze *exact* in output: camera din
`room class`, dispozitivul din `name`, clasa din `class`, actiunea din `Action name`, clasa actiunii din `command class`.

### 1.3 Output-ul exact

Un obiect JSON (modelul il scrie dupa ce a "gandit" in `<think>...</think>`), citit tolerant (`loose_json_loads`) si
transformat in `GoalStructure` (`ami_agents/shared/models/goal_structure.py`). Pentru un goal explicit sau incomplet:

| camp | ce spune | exemplu (cererea "Set the air conditioner temperature to 20 degrees in the guest bedroom") |
|---|---|---|
| `goal_specificity` | explicit (un dispozitiv si o actiune) / incomplete (lipseste ceva) / ambiguous (doar o dorinta: "e prea cald") | `explicit` |
| `location_class` | camera | `homeont:GuestBedroom` |
| `artifact_class` | tipul dispozitivului | `homeont:AirConditioner` |
| `artifact_name` | dispozitivul anume | `guest_bedroom_air_conditioner` |
| `device_reference` | cererea il numeste ("name") sau il descrie ("description") | |
| `affordance_class` | clasa comenzii | `saref:SetAbsoluteLevelCommand` |
| `affordance_name` | numele actiunii | `temperature` |
| `parameter_name` | parametrul actiunii; `"NA"` = nu are, `null` = trebuie dar nu se stie | `NA` |
| `goal_effect` | `set` (valoare absoluta) / `modify` (schimbare relativa) | `set` |
| `target_value_text`, `target_value_determined` | cuvintele utilizatorului / valoarea in unitatea actiunii | `20 degrees` / `20.0` |
| `quantifier` | pentru goal-uri incomplete: toate dispozitivele sau unul | `null` |
| **`unmatched_device`** | cuvintele utilizatorului pentru un dispozitiv **care nu e in casa** ("air conditioner") | `null` |
| **`unmatched_action`** | idem pentru o setare pe care dispozitivul nu o are ("brightness" pe un bec on/off) | `null` |

Un goal ambiguu nu numeste dispozitiv, ci `implied_environment_vars` (variabile de mediu si sensul dorit: "so damp" =
umiditate, `decrease`). Un `dependency` adauga `timing` (cand incepe) si predicate (conditii).

Doua detalii importante pentru tot ce urmeaza:
- `GoalStructure.from_dict` e *tolerant*: ignora chei necunoscute, transforma `"NA"/"null"` in `None`, dar **nu verifica ca un
  IRI exista in casa**. Un nume inventat ajunge nemodificat la planificator.
- `unmatched_device`/`unmatched_action` sunt iesirile prin care structurarea spune "nu exista". Un goal cu `unmatched_device`
  nu e planificat: `goal_resolution.unmatched_question` il transforma in "I can't find the X in the Y. What should I do
  instead?" (`goal_resolution.py`, `clarification_question`).

### 1.4 Ce trebuie sa faca, intr-o fraza

Sa transforme o cerere in limbaj natural intr-o **descriere formala ancorata in vocabularul exact al casei**, in care
camera, dispozitivul, actiunea si valoarea sunt coerente intre ele (dispozitivul e in camera aceea, actiunea e a dispozitivului
acela, valoarea are unitatea actiunii), si sa **semnaleze explicit** cand cererea nu poate fi servita de aceasta casa.

Ce se intampla dupa, si de ce conteaza exactitatea sirurilor: planificatorul explicit cauta in graful casei dispozitivul cu
clasa si camera din goal si actiunea cu clasa data (taxonomia de erori spune ca "planificatorul compara sirurile claselor").
O clasa fara prefix (`DimmableLight`) nu se potriveste cu `homeont:DimmableLight`.

### 1.5 Cum se scoreaza: de ce sunt doua sarcini

(`scoring.py`, legenda din `summary.md`.) Un episod trece daca **fiecare parte gold e indeplinita**:

- parte **fezabila**: o actiune acceptata de casa care e exact apelul gold (cu valoarea exacta);
- parte **infezabila** (`error_input` in gold): un goal raspuns "impossible" sau cu o intrebare (un refuz).

Deci aceeasi structurare trebuie sa fie *precisa* pe cererile fezabile si *refuzanta* pe cele care nu au sens in casa. Intreaga
poveste a experimentelor de mai jos e tensiunea dintre aceste doua cerinte.

Compozitia ECAI-400 (din `errors_episodes.csv`): 114 episoade `hb1` fezabile (o comanda), 86 `hb1` infezabile (64 `unexist_device`,
22 `unexist_attribute`), 104 fezabile cu mai multe parti (`hb2`-`hb8`), 96 `mixed` (parti fezabile si infezabile in acelasi episod).
Atentie: dosarul contine o analiza care arata ca multe etichete `unexist_attribute` contrazic casa (un bec care are `brightness`
e etichetat ca neavand-o); rata pe infezabile are deci un zgomot de eticheta (codul `infeas_accepted_on_target`, 10-12 parti in G/G2).

### 1.6 Exemplu complet (M3, un episod)

Cererea: "Set the brightness of the light to 10 in the master bedroom" (`hb1_feasible_seed_69_fixed`). Segmentarea da un goal
`explicit, achievement`. Raspunsul real al lui M3 (dupa ~800 de tokeni de gandire):

```json
{"goal_specificity": "explicit", "goal_kind": "achievement",
 "intent_text": "Set the brightness of the light to 10 in the master bedroom",
 "location_class": "Master bedroom", "artifact_class": "DimmableLight",
 "artifact_name": "master_bedroom_light", "device_reference": "name",
 "affordance_class": "SetAbsoluteLevelCommand", "affordance_name": "brightness",
 "parameter_name": "NA", "goal_effect": "set", "target_value_text": "10",
 "target_value_determined": "10", ...}
```

Modelul a inteles totul corect: camera, dispozitivul, actiunea, valoarea. Dar a scris `Master bedroom` in loc de
`homeont:MasterBedroom`, `DimmableLight` in loc de `homeont:DimmableLight`, `SetAbsoluteLevelCommand` in loc de
`saref:SetAbsoluteLevelCommand`. Planificatorul nu gaseste nimic si raspunde "impossible". Asta este problema care a pornit
experimentele.

---

## 2. Varianta implicita (M3): decodare libera

### 2.1 Ce face

`goal_structuring_variants.free_decoding`: **un singur apel**, `chat.completions.create(model, messages, temperature=0)`.
Modelul (un model de rationament) isi scrie gandirea, apoi JSON-ul; se ia `message.content`, se parseaza, se construieste
`GoalStructure`. Nimic nu verifica numele. "Toate axele pe valoarea implicita" este apelul vechi **octet cu octet** (un test
fixeaza hash-ul promptului; controlul M3 rerulat pe un episod da aceeasi structura si aceiasi tokeni).

### 2.2 De ce pica (cifre din rularea fep, 400 de episoade)

| indicator M3 | valoare |
|---|---|
| episoade trecute | 90 din 399 scorate = **22,6%** |
| hb1 fezabile / hb1 infezabile | **7,9%** / **88,2%** |
| parti fezabile trecute | 91 din 734 = **12,4%** |
| parti infezabile trecute | 245 din 256 = **95,7%** |
| parti picate, pe etapa | structurare 632, planificare 13, executie 4, segmentare 2, infra 2, fezabilitate 1 |
| codurile de eroare dominante | `struct_class_prefix` 234, `struct_slot_confusion` 120, `struct_wrong_value` 89, `struct_wrong_room` 82, `struct_invalid_device_name` 38 |

Concluzia citita din date: din 654 de parti picate **632 se pierd la structurare**. Cea mai mare cauza (`struct_class_prefix`, 234) e
notatie: modelul alege bine, scrie gresit. A doua (`struct_slot_confusion`, 120): o valoare in campul gresit (o clasa de
dispozitiv in `location_class`). Apoi: camera gresita, nume de dispozitiv inventat.

Infezabilele "trec" la 95,7% pentru un motiv gresit: numele stricate nu gasesc dispozitivul, deci planificatorul raspunde
"impossible" la aproape orice (pe episoadele `mixed`, 99% din partile infezabile sunt refuzate, dar si partile fezabile sunt refuzate: 1,87 refuzuri per cerere pe `hb2` mixed). Un M3 care ar scrie numele corect ar pierde aceste
treceri. Deci scorul M3 pe infezabile **nu e o masura a abilitatii de a spune "nu exista"**; e un efect secundar al erorilor.

### 2.3 Axele configurabile (cum se fac experimentele)

Mecanismul (`shared/utils/prompt_variants.py`, `goal_structuring_variants.py`): pasul are *axe*, fiecare cu *optiuni* numite; un
fisier de varianta alege cate o optiune pe axa (`prompt_variants: structuring: {decoding: iri_trie}`).

| axa | implicit | optiuni | ce schimba |
|---|---|---|---|
| `context` | `text` | `json`, `json_v2` | formatul contextului si versiunea regulilor |
| `prompt` | `full` | `focused` | promptul scurtat + 2 exemple |
| `constraint` | `none` | `schema` | `response_format` json_schema strict (enumuri plate) |
| `decoding` | `free` | `iri_trie`, `genre`, `iri_trie_v2`, `genre_v2` | cum se aleg tokenii raspunsului |
| `repair` | `none` | `class_names` | dupa parsare, rescrie clasele cum le scrie casa |

Bratele: A = M3 (control, rulat), B schema, C json, D focused, E repair, F json_v2, **G `iri_trie`**, **H `genre`**, **G2
`iri_trie_v2`**, **H2 `genre_v2`**. Rulate pe cele 400 pe fep pana acum: M3, G, H, G2 (si json-v2, in alt raport); H2 a rulat dupa scrierea acestui raport (vezi `experimente_complet.md`; nu e in tabelele de mai jos).
Fisierele: `tests/homebench/variants/homebench.ecai.M3.pv-*.yaml`; ele difera de M3 doar prin `prompt_variants:` si, la H/H2,
prin providerul structurarii.

---

## 3. Varianta trie (G, `decoding: iri_trie`)

### 3.1 Ideea (comuna cu H)

GENRE (De Cao et al., "Autoregressive Entity Retrieval", ICLR 2021, citat in docstring-ul `iri_trie.py`; nu l-am verificat pe sursa
originala acum) scrie numele unei entitati token cu token si, la fiecare pas, permite doar tokenii care mentin textul scris intr-un
**trie** (arbore de prefixe) al numelor valide. Rezultat: ce se scrie exista, garantat. Aici "entitatile" sunt valorile
sloturilor care numesc ceva din casa (`SLOT_VOCABULARY`): `location_class`, `artifact_class`, `artifact_name`, `affordance_class`,
`affordance_name`, `parameter_name`, `property_*`, `space_*`. Se constrange **doar interiorul valorii dintre ghilimele**; cheile,
gandirea, textul liber si `null` raman libere.

Trie-ul unui slot nu e plat; e **conditionat de ce contine deja acelasi obiect** (`iri_paths`, `slot_values`): casa e transformata in
randuri (un rand per actiune, proprietate, variabila de mediu si camera); dupa `location_class = homeont:MasterBedroom` valorile
permise pentru `artifact_name` sunt dispozitivele *acelei* camere; dupa dispozitiv, `affordance_name` poate fi doar o actiune a lui.
Daca niciun rand nu se potriveste, se foloseste vocabularul plat al slotului (`conditioned: false`), ca raspunsul sa ramana mereu posibil.
Trie-ul e pe **caractere**, nu pe id-uri de tokeni, deci orice tokenizare a unui nume valid e acceptata.

### 3.2 De ce o emulare

Ollama nu da acces la logits si nu accepta o gramatica pe un mesaj continuat. Arata cel mult 20 de tokeni pe pas
(`top_logprobs=20`). Deci nu se poate masca vocabularul; algoritmul `decode_with_iri_trie` face altceva:

1. **Apelul de control**: raspunsul se genereaza liber (greedy), exact ca in M3.
2. **Verificare**: fiecare valoare de slot din raspuns se cauta in trie-ul ei. O valoare valida **se pastreaza fara niciun apel
   in plus** (decodarea constransa greedy ar fi scris exact aceeasi valoare: fiecare prefix era permis).
3. **Prima valoare invalida** (`first_invalid_slot`): raspunsul se taie la ghilimeaua de deschidere. Valoarea se redecodeaza **un token
   pe apel**: raspunsul de pana atunci e trimis ca mesaj de asistent neterminat, `max_tokens=1`, `top_logprobs=20`; din cei 20
   de tokeni se ia **cel mai probabil dintre cei permisi de trie**. Caracterele pe care trie-ul le forteaza (o singura
   ramura) se scriu fara apel.
4. **Inchidere**: valoarea se inchide cu tokenul de inchidere al modelului (`",` sau `"}`), nu cu un `"` gol (altfel modelul crede ca
   a scris deja virgula si strica JSON-ul; un bug gasit in smoke test).
5. **Continuare** (`resume`): modelul isi continua liber raspunsul de la valoarea corectata si verificarea merge la urmatoarea
   valoare invalida (maxim 24 de interventii pe raspuns, 24 de pasi pe valoare).
6. **`nearest`**: daca niciunul dintre cei 20 de tokeni nu e permis, se lasa modelul sa continue liber valoarea si textul rezultat
   se aliniaza la cea mai apropiata valoare din trie (similaritate de caractere). Se logheaza (`how: nearest`; G: 56 de cazuri, G2: 18).
7. **Regula evidentei** (`narrow_by_evidence`, D58, doar in G2 ca rulare pe 400): trie-ul cu care se redecodeaza o valoare *invalida*
   se restrange si cu sloturile scrise de model *dupa* ea in acelasi obiect, daca acestea exista in casa si lasa mai putine valori.
   Motivul: in `hb1_feasible_seed_4351` modelul scrisese `affordance_class: homeont:TemperatureSetpoint` (invalid) apoi
   `affordance_name: temperature` (corect); greedy a luat ramura `homeont:`, a scris `homeont:SetModeCommand`, iar `temperature`
   a devenit `mode`: trie-ul isi trata propria ghicire ca fapt si suprascria o valoare corecta.

### 3.3 Exemplu: episodul 4351 prin G2 si M3

Cererea: "Set the air conditioner temperature to 20 degrees in the guest bedroom." Gold: `guest_bedroom.air_conditioner.set_temperature(20)`.

| | M3 | G | H | G2 |
|---|---|---|---|---|
| `location_class` | `GuestBedroom` | `homeont:GuestBedroom` (corectat) | `homeont:GuestBedroom` (masca) | `homeont:GuestBedroom` (corectat) |
| `affordance_class` | `SetAbsoluteLevelCommand` | `saref:SetAbsoluteLevelCommand` | `saref:SetAbsoluteLevelCommand` | `saref:SetAbsoluteLevelCommand` |
| `parameter_name` | `temperature` | `NA` (corectat) | null | null |
| rezultat | **pica** (`struct_class_prefix`) | trece | trece | trece |

In G2 singura interventie e `GuestBedroom -> homeont:GuestBedroom` (1 candidat, 0 pasi de model): atat a fost de reparat.

### 3.4 Ce comportamente da

- **Remediaza notatia complet**: `struct_class_prefix` 234 -> 0 si `struct_invalid_device_name` 38 -> 0 (G, H si G2 toate). Parti fezabile
  trecute 12,4% -> 55,3% (G).
- **Numele sunt mereu din casa, chiar daca gresite ca sens**: valoarea e valida, dar poate fi alta decat cea cerute (urmeaza).
- **Costul** (doar pasul de structurare, medie pe episod): G are 16,2 apeluri LLM (M3: 2,5), 95.500 de tokeni de intrare (promptul se retrimite la
  fiecare apel; M3: 11.800; dupa documentatia bratului, mare parte e servita din cache-ul de prompt al Ollama), 1.885 de tokeni de iesire (M3: 1.278). G2: 12,4 apeluri,
  68.300 tokeni de intrare. Timpii de perete **nu se pot compara** intre rulari: M3 si G au rulat cu `workers: 4`, H si G2 cu `workers: 1`
  (`run.json`); de aceea G (37 s structurare) si G2 (9,9 s) difera mult cu aceeasi metoda.
- **Limite ale emularii**: alegere greedy (un prefix de namespace ales gresit nu se mai poate corecta decat prin evidenta), doar
  20 de candidati pe pas, `nearest`.

### 3.5 Slabiciunea cheie: G inlocuieste in loc sa refuze

Exemplu real, `hb1_infeasible_seed_4644`: "Increase the intensity of the humidifier in the store room by 50%." Gold: `error_input`
(camera nu are umidificator).

| | M3 | G | H | G2 |
|---|---|---|---|---|
| ce scrie raspunsul liber | `StoreRoom`, `Humidifier`, `store_room_humidifier` | `Humidifier` pus in `location_class` (slot confundat), `homeont:Humidifier`, `store_room_humidifier`, `intensity` | (raspunsul liber nu e in tabelele de aici) | `StoreRoom`, `homeont:Humidifier` |
| structura finala | nume fara prefix, nu se gaseste nimic | `homeont:Dehumidifier`, `store_room_dehumidifiers`, `onOff` (6 interventii) | `store_room_dehumidifiers`, `intensity` | `unmatched_device` = "humidifier" |
| rezultat | refuz, **trece** (din greseala) | executa pe dezumidificator, **pica** (`infeas_substituted`) | pica | refuz corect: "I can't find the humidifier in the store room.", **trece** |

Trie-ul oferea o valoare pentru fiecare slot, deci `Humidifier` (o clasa reala a ontologiei, dar absenta din camera) a fost inlocuit cu cea
mai apropiata clasa pe care camera o are. Din 39 de valori `artifact_class` inlocuite la episoadele hb1 infezabile din G, 36 erau un tip real
de dispozitiv absent din camera (la fezabile: 2 din 21). Deci modelul citise bine cererea, iar constrangerea a suprascris-o.

---

## 4. Varianta GENRE (H, `decoding: genre`)

### 4.1 Ce e diferit

Aici modelul ruleaza **in procesul unui server** (`tests/simuhome/bench/genre_server.py`, endpoint compatibil OpenAI, port 11600) cu
`transformers`, ponderi HuggingFace bf16 (necuantizate, nu cele Q4_K_M din Ollama), pe un H200. Doar structurarea merge la
server; restul pasilor raman pe Ollama. Un apel de structurare face **trei generari** pe acelasi GPU:

1. **gandirea** (`<think>...`), libera, greedy, pana la `</think>` (oprire prin `ThinkEnd`);
2. **raspunsul liber** din aceeasi gandire (controlul pe acest backend; se salveaza ca `free_answer`);
3. **raspunsul prin beam search** (`num_beams=4`, `length_penalty=1.0`, max 2048 de tokeni) sub un `GenreProcessor`
   (`LogitsProcessor`).

### 4.2 Masca (`GenreProcessor` + `genre_decoding.py`)

La fiecare pas si pentru fiecare fascicul se citeste textul scris pana atunci:

- **in interiorul valorii unui slot constrans**: doar id-urile de tokeni care tin valoarea in trie-ul slotului (`HomeRules.next_tokens`)
  pastreaza scorul, restul primesc `-inf`. Un token poate si inchide valoarea (`"`, `",`, `"}`) daca e completa. `TokenTrie` e un trie pe
  caractere peste textul fiecarui token din vocabular (151.665 de tokeni), deci orice tokenizare a unui nume e acceptata.
- **in afara unei valori** (chei, text liber, `null`): doar tokenii care pot continua un obiect JSON (`JsonPrefix`, automat pe
  caractere), iar tokenul de sfarsit de secventa doar cand raspunsul e deja un obiect JSON complet (`may_end`).

Cele doua reguli JSON exista pentru ca beam search compara **secvente intregi**: fara `may_end`, cea mai ieftina secventa se opreste
imediat; fara `JsonPrefix`, un fascicul care iesea din trie prin `null` scria apoi un comentariu `// ...` si ajungea la plafon
(`hb1_infeasible_seed_4644`, D60). `null` ramane o iesire legitima.

Ce elimina fata de G: limita de 20 de tokeni, `nearest`, alegerea greedy a namespace-ului, un apel pe token, problema `"` vs `",`.
In schimb: un backend diferit (ponderi bf16 + alt template de chat), asa ca **M3 nu e un control curat pentru H**; controlul curat e `free_answer`.
O rulare `decoding: free` pe HF pe cele 400 ar separa cele doua efecte; nu exista.

### 4.3 Ce a dat H pe fep (400 de episoade, job 300297)

- Cel mai bun pe hb1 fezabile: **64,9%** (G 51,3%; M3 7,9%) si pe partile fezabile (59,4%).
- Cel mai slab pe infezabile: **32,6%** pe hb1 (G 48,8%), partile infezabile 33,5%.
- Trei cauze, citite din inregistrari (nu tin de constrangerea GENRE):
  1. **`null` ca iesire ieftina** pentru beam search: un `null` costa cativa nats, un IRI lung mai mult. In H: 88 de goal-uri cu *toate* sloturile
     IRI null (la G: 8) si alte 150 fara camera (G: 19). Un goal total-null e `unrouted` in planificator si nu e un refuz. Efect: `seg_part_lost` 62
     (M3: 2), `struct_room_missing` 70 (G: 10).
  2. **Chei cu cratima**: beam a scris `"location-class"` in loc de `"location_class"`; masca recunoaste doar numele cu underscore, deci nimic nu a fost
     constrans si raspunsul nu s-a parsat: 15 episoade cu "no goal could be structured".
  3. **Gandire taiata la plafon**: 3 erori de infrastructura (endpoint fara inregistrare `genre`).

---

## 5. Imbunatatirile adaugate si ce comportament da fiecare

| imbunatatire | unde in cod | de ce a fost adaugata | ce comportament da | in ce brat |
|---|---|---|---|---|
| trie conditionat pe sloturile deja scrise | `iri_trie.iri_paths`, `slot_values` | un dispozitiv trebuie sa fie din camera scrisa, o actiune din dispozitivul scris | valori coerente intre ele; fallback la vocabularul plat daca niciun rand nu se potriveste | G, H, G2, H2 |
| inchidere cu tokenul modelului (`",`) | `_Decoder._closing` | un `"` gol facea modelul sa sara virgula | JSON valid dupa corectie | G, G2 |
| `parameter_name` constrans, cu `NA` | `EXTRA_VALUES`, `flat_values` | casa fara parametri lasa modelul sa scrie un parametru inventat | `parameter_name: "NA"` pentru actiuni cu valoare directa (cea mai frecventa interventie dupa camera: 538 in G) | toate |
| regula evidentei (D58) | `narrow_by_evidence` | 4351: o clasa ghicita strica o actiune corecta | `struct_wrong_affordance` 64 -> 12 parti; re-decodari `affordance_name` 235 -> 68; mai putine apeluri | G2 (prima rulare pe 400) |
| trace complet al decodarii | `TrieDecoding.trace`, `decoding_report.py` | sa se vada fiecare decizie | `structures[].decoding.trace`; raport pas cu pas | G2 |
| `may_end` + `JsonPrefix` | `genre_decoding.py` | beam: raspuns de 1 token / comentarii `//` | masca JSON; `null` ramane permis | H, H2 |
| un singur thread pe model, `ThinkEnd`, watchdog | `genre_server.py`, `slurm_harness.sbatch` | scurgere de ~1 MB RAM per token pe thread; serverul murea la ~34 de episoade | rulare stabila pe 400 | H, H2 |
| **abstinenta** pe `artifact_class` | `abstention.py`, parametrul `abstain` | G/H inlocuiau un dispozitiv absent | un `artifact_class` care e familie de dispozitive in `homeont.ttl`, absent din camera numita si fara clasa din aceeasi familie (ultimul cuvant CamelCase) nu se mai decodeaza: obiectul devine `unmatched_device` = cuvintele utilizatorului, sloturile IRI null -> "I can't find the X" | G2, H2 |
| `no_null` | `HomeRules.token_filter` | beam alege `null` | fara token `null` pentru camera, dispozitiv, actiune cat timp casa are valori | H2 |
| `keys` | `HomeRules.keys` | chei cu cratima | o cheie JSON ramane prefix de cheie din raspunsul liber | H2 |
| `close_thinking` | `genre_server._close_thinking` | gandire taiata = eroare | se inchide cu `</think>` si se raspunde din ea | H2 |
| abstinenta citita din raspunsul liber | `abstention.overlay` | sub masca, modelul nu mai poate scrie dispozitivul absent | se suprapune peste acelasi goal din raspunsul beam, pereche pe pozitie | H2 |

Regula abstinentei e ingusta intentionat: doar `artifact_class`, doar cand valoarea e o clasa-dispozitiv a ontologiei (o valoare garbled e o
greseala de scriere pentru trie), si doar cand nici familia ei nu e in camera (`DimmableLight` ceruta intr-o camera cu `ColorLight` se
repara, nu se refuza). Abstinenta pe actiuni (`unmatched_action`) nu e facuta, deliberat: 3 din 15 structuri fezabile hb1 cu ambele sloturi de
actiune invalide trec dupa inlocuirea de trie, fata de cel mult 2 infezabile castigate.

---

## 6. Rezultate pe fep (400 de episoade, aceleasi episoade, aceleasi modele)

Rulari: M3 job 294585 (commit `bfb099d4`), G job 299621 (`bf3be605`), H job 300297 (`bf3be605`), G2 job 300435 (`22ad219c`); toate pe H200.

### 6.1 Succes pe episoade (toate partile gold indeplinite; infra scoase)

| grup | n | M3 | G | H | **G2** |
|---|---|---|---|---|---|
| toate | 399 | 90/399 = 22,6% | 135/394 = 34,3% | 127/397 = 32,0% | **172/393 = 43,8%** |
| hb1 fezabile | 114 | 9/114 = 7,9% | 58/113 = 51,3% | 74/114 = 64,9% | **74/112 = 66,1%** |
| hb1 infezabile | 85 | 75/85 = 88,2% | 41/84 = 48,8% | 28/86 = 32,6% | 49/84 = 58,3% |
| fezabile cu mai multe parti | 104 | 0/104 = 0,0% | 28/102 = 27,5% | 21/102 = 20,6% | 33/102 = 32,4% |
| mixed | 96 | 6/96 = 6,2% | 8/95 = 8,4% | 4/95 = 4,2% | 16/95 = 16,8% |

(Diferente mici fata de `findings.md`, de ex. hb1 fezabile G2 74/112 aici, 73/111 acolo, vin din ce episoade se numara ca infra; nu schimba concluziile.)

### 6.2 Parti

| | M3 | G | H | G2 |
|---|---|---|---|---|
| parti fezabile trecute | 91/734 = 12,4% | 398/720 = 55,3% | 429/722 = 59,4% | **463/720 = 64,3%** |
| parti infezabile trecute | **245/256 = 95,7%** | 143/250 = 57,2% | 85/254 = 33,5% | 163/252 = 64,7% |
| parti picate (total) | 654 | 429 | 462 | **346** |

### 6.3 Perechi (acelasi episod, doua rulari)

| pereche | n | castigate | pierdute | trecute A | trecute B |
|---|---|---|---|---|---|
| M3 -> G | 393 | 87 | 40 | 88 | 135 |
| M3 -> H | 396 | 94 | 57 | 90 | 127 |
| M3 -> G2 | 393 | 116 | 33 | 89 | 172 |
| G -> G2 | 388 | 64 | 30 | 135 | 169 |
| H -> G2 | 393 | 84 | 39 | 127 | 172 |
| G -> H | 392 | 52 | 63 | 135 | 124 |

### 6.4 Coduri de eroare (parti picate)

| cod | M3 | G | H | G2 | ce inseamna |
|---|---|---|---|---|---|
| `struct_class_prefix` | 234 | 0 | 0 | 0 | clasa fara prefix |
| `struct_slot_confusion` | 120 | 0 | 0 | 1 | valoare in campul gresit |
| `struct_wrong_value` | 89 | 119 | 79 | 126 | valoarea tinta difera de gold |
| `struct_wrong_room` | 82 | 6 | 1 | 1 | alta camera decat gold |
| `struct_wrong_affordance` | 18 | 64 | 22 | 12 | alta actiune decat gold |
| `struct_room_missing` | 11 | 10 | **70** | 7 | goal fara camera |
| `seg_part_lost` | 2 | 9 | **62** | 14 | nicio structura pentru parte |
| `exec_wrong_value` | 4 | 59 | 54 | 47 | apelul corect cu alta valoare |
| `plan_unrouted` | 8 | 23 | 41 | 11 | planificatorul nu accepta structura |
| `plan_explicit_failed` | 1 | 30 | 25 | 18 | planificatorul explicit a esuat |
| `plan_false_impossible` | 0 | 19 | 15 | 21 | planificatorul zice "imposibil" pentru fezabil |
| `infeas_substituted` | 1 | **33** | 16 | 17 | s-a operat alt dispozitiv in loc sa se raporteze lipsa |
| `infeas_executed_other` | 0 | 10 | 14 | **21** | nimic refuzat, actiuni executate in alta parte |
| `infeas_accepted_on_target` | 0 | 12 | 11 | 10 | casa a acceptat actiunea pe dispozitivul "imposibil" (eticheta suspecta) |
| `struct_false_clarify_device` | 1 | 2 | 4 | **14** | "can't find X" pentru un dispozitiv care exista |
| `struct_json_invalid` | 2 | 6 | 0 | 6 (10 structuri cu eroare de parsare) | raspuns care nu e un obiect JSON |
| `infra_llm_error` | 0 | 0 | 16 | 0 | |

Citire: **notatia (`struct_class_prefix`, `struct_slot_confusion`, `struct_invalid_device_name`) dispare** in toate cele trei brate cu
decodare; ce ramane (`struct_wrong_value`, `exec_wrong_value`) e valoarea, nu numele. G2 curata `struct_wrong_affordance` (64 -> 12) si
`infeas_substituted` (33 -> 17), dar adauga `infeas_executed_other` (10 -> 21) si `struct_false_clarify_device` (2 -> 14).

### 6.5 Ce au facut decodarile, per apel de structurare

| | apeluri cu inregistrare | apeluri LLM totale | valori redecodate/schimbate | cum | abstinente |
|---|---|---|---|---|---|
| G | 955 | 6386 | 2329 (camera 868, `parameter_name` 538, `affordance_class` 332, `affordance_name` 235, `artifact_name` 191, `artifact_class` 162) | trie 2273, nearest 56 | - |
| G2 | 957 | 4871 | 1932 (camera 860, `parameter_name` 485, `affordance_class` 279, `artifact_class` 119, `artifact_name` 113, `affordance_name` 68) | trie 1846, nearest 18, abstain 68 | 68: 46 in `mixed`, 19 in infezabile, **3 in fezabile (false)** |
| H | 960 | - | 2006 de sloturi schimbate fata de raspunsul liber (camera 623, `artifact_class` 593, `affordance_class` 533) | masca | (H nu are abstinenta) |

Abstinenta: 65 din 68 sunt pe un episod cu parte infezabila. Cele 3 false sunt aspiratorul (`homeont:Rvc`), vezi 7.3.

### 6.6 Costul rularii

Structurare, medie pe episod (400 de episoade): M3 18,9 s si 2,5 apeluri; G 37,0 s si 16,2 apeluri; H 35,0 s si 2,4 apeluri (3 generari pe server);
G2 9,9 s si 12,4 apeluri. Timpii depind de `workers` (4 la M3/G, 1 la H/G2): nu se compara intre grupe; **numarul de apeluri** si de tokeni se compara.

### 6.7 Cat de mult se poate avea incredere in diferente (verificat in acest raport)

- **Zgomotul rularii**: reconstruind raspunsul liber din trace-ul lui G2 si comparandu-l cu structura din M3 (acelasi model, acelasi prompt,
  `temperature: 0`), doar **50%** dintre cele 109 episoade hb1 fezabile au aceleasi sloturi `artifact_class`, `artifact_name`, `affordance_class`,
  `affordance_name`, `goal_effect`, `target_value_determined` (diferentele: valoare 30, clasa 24, `goal_effect` 15). Comparatia e indicativa (structura M3
  e dupa rezolvare, nu textul brut), dar spune ca doua rulari ale "aceleiasi" metode difera si ca o parte din cele 64 castigate / 30 pierdute G -> G2 e zgomot,
  nu metoda. Nu am o masura curata a zgomotului: raspunsul liber al lui G v1 nu e salvat (codul din `bf3be605` nu avea trace).
- **G -> G2 amesteca doua schimbari** (evidenta + abstinenta); nu exista un brat doar cu una.
- **H vs M3**: backend diferit (ponderi, template). Controlul curat e `free_answer` din inregistrarea lui H.
- Eticheta infezabilelor `unexist_attribute` e suspecta in parte (vezi 1.5); episoadele infra (2-4 per rulare) sunt scoase.

---

## 7. Ce a ramas de rezolvat, in ordinea in care conteaza (cifre din G2)

Eticheta: **[verificat]** = numarat in inregistrari in acest raport; **[ipoteza]** = explicatie plauzibila, nemasurata.

### 7.1 Cererile relative: `goal_effect: set` in loc de `modify` [verificat]

La hb1 fezabile, cereri relative ("Increase ... by 34%", "Lower ... by 5 degrees"; clasificate cu un regex simplu, deci aproximativ):

| rulare | `goal_effect` scris | n | trec |
|---|---|---|---|
| M3 | modify / set | 16 / 26 | 0 / 0 |
| G | modify / set | 19 / 24 | 11 (57,9%) / **0** |
| H | modify / set | 20 / 20 | 13 (65%) / 1 |
| G2 | modify / set | 22 / 20 | 15 (68,2%) / **0** |

Modelul scrie `set` cu valoarea 34 (un delta), iar gold-ul e absolut (curent + 34), deci planificatorul seteaza 34. Cererile absolute cu
`set` trec 89% (G2). Aceasta este cea mai mare pierdere ramasa pe hb1 fezabile (20 din cele 38 de episoade picate). Nu e o problema de decodare; e
o regula deterministica posibila (`set -> modify` cand cererea are "increase/lower/... by N"), propusa in `findings.md` ca brat separat pe axa `repair`.

### 7.2 Regula evidentei schimba camera corecta [verificat; cauza din trace, remedierea e propunere]

In G2, **36 de episoade** au o camera pe care raspunsul liber o scrisese corect (doar fara prefix: `GuestBedroom`, `Master Bedroom`) inlocuita de
decodare cu **alta camera** (G: 9). **31 din cele 36 pica** (primul cod: `infeas_executed_other` 9, `struct_wrong_value` 9). Mecanismul, vazut in trace:

```
hb1_infeasible_seed_2595: "Set the air conditioner in the guest bedroom to auto mode for swing."
  location_class liber: GuestBedroom   (valid, doar fara prefix)
  evidenta (affordance_name = swing) -> trie permis: ['homeont:LivingRoom']   (singura camera cu o actiune `swing`)
  -> location_class = homeont:LivingRoom, dispozitiv `living_room_air_conditioner`: comanda se executa in alta camera
hb1_infeasible_seed_4138: "Set the volume of the media player to 40 in the master bedroom."
  evidenta (affordance_name = volume) -> ['homeont:Balcony', 'homeont:Garage', 'homeont:LivingRoom']  -> LivingRoom
```

`location_class` e primul slot si nu are nimic scris inainte; regula evidentei il restrange la camerele care au actiunea scrisa dupa el. Daca
dispozitivul cerut lipseste din camera ceruta (un caz infezabil), camera ceruta iese din trie si e inlocuita cu o camera care il are; apoi
abstinenta compara dispozitivul cu *noua* camera (care il are), deci nu se mai declanseaza. Asta explica `infeas_executed_other` 10 -> 21. Camera e informatia
cea mai sigura din cerere si nu ar trebui sa fie suprascrisa de o actiune. Remediere propusa (nemasurata): daca valoarea scrisa de model, normalizata (fara prefix,
spatii, litere mari), se potriveste cu exact un candidat din trie-ul *fara* evidenta, se alege acela (repair inainte de trie), si evidenta nu mai poate scoate
din trie valoarea pe care modelul a scris-o deja.

### 7.3 Abstinenta fals pozitiva [verificat]

- **Aspiratorul**: 3 episoade fezabile (`hb1_feasible_seed_5341`, `5478`, `hb5_feasible_seed_456_fixed`) cer `vacuum_robot.set_cleaning_area(<camera>)`.
  Modelul scrie `location_class` = camera de curatat, `artifact_class: homeont:Rvc`; camera aceea nu are un aspirator (aspiratorul e in alta parte; camera
  e *argumentul* comenzii), deci regula declanseaza "I can't find the rvc in the dining room". Regula presupune ca `location_class` = unde sta dispozitivul.
- Cuvintele pentru utilizator se fac din CamelCase (`class_words`): `homeont:HUMIDIFIER` (scris cu majuscule de model) devine "h u m i d i f i e r".
  Cosmetic: intrebarea ramane un refuz, dar textul e urat (1 episod, `hb8_mixed_seed_334_fixed`).

### 7.4 Substitutia facuta de model, nu de trie [verificat]

Din cele 35 de episoade infezabile picate in G2, 10 sunt `infeas_substituted`; in mai multe, **raspunsul liber** a scris deja alt dispozitiv:
"humidifier" -> `Dehumidifier` (1579, 2569, 4532, 5751; pentru 1579 si 2569 analiza din 16.11 noteaza ca dezumidificatorul exista in camera), "heating" -> `AirConditioner`/`Heater` (3386, 4506, 5726). Regula
abstinentei nu are ce sa faca: valoarea e o clasa prezenta in camera. Se rezolva doar daca dispozitivul din goal se verifica fata de cuvintele cererii
(etichetele ontologiei, `skos:altLabel`); propus in `findings.md`.

### 7.5 JSON duplicat dupa o corectie [verificat pe 2 din 10 cazuri; mecanismul pentru restul e ipoteza]

G2 are 10 structuri cu eroare "expected a JSON object" (G 7, M3 2). In `hb1_infeasible_seed_1701` si `4786`, dupa valoarea corectata modelul continua cu
un comentariu `// ...` sau reia cheile obiectului, apoi alt `resume` o ia de la capat: textul final contine obiectul de mai multe ori (`resume` repetat de 4 ori)
si nu se parseaza. Aceste episoade pierd structura si apar ca `struct_json_invalid`/`seg_part_lost`. H il evita prin `JsonPrefix` (mascat), G/G2 nu au un
echivalent al masti JSON in continuarea libera.

### 7.6 Ce nu am masurat

- **H2** (genre_v2): la scrierea acestui raport nu rulase; a rulat pe 400 (job 302815, 41,3%, vs G2 43,8%, p = 0,32) si repara infezabilele lui H (hb1: 32,6 -> 59,3%).
  Gaura ramasa (cheile JSON omise de beam, 11,4% din apeluri) e in `experimente_complet.md` 8.5; tabelele de aici nu au fost refacute cu H2.
- Un control `decoding: free` pe backend-ul HF.
- Efectul separat al evidentei si al abstinentei (cele doua sunt in G2 impreuna).
- `plan_unrouted` / `plan_explicit_failed` (30 / 25 de parti la G / H, 18 la G2): planificatorul, nu structurarea.

---

## 8. Cum reproduci si verifici

```bash
# tabelele acestui raport (din fisierele deja existente, fara LLM)
python -I work/reports/structuring_analysis.py            # din radacina repo-ului; foloseste tests/homebench/results/harness/

# rapoartele de erori pe o rulare
python -m tests.homebench.error_report tests/homebench/results/harness/fep.trie-v2

# o decizie de decodare desenata pas cu pas (episod)
python -m tests.homebench.decoding_report --config tests/homebench/variants/homebench.ecai.M3.pv-trie-v2.yaml --episode hb1_infeasible_seed_4644

# H2 pe fep (cere GPU + ponderi HF)
sbatch --export=ALL,VARIANT=tests/homebench/variants/homebench.ecai.M3.pv-genre-v2.yaml,GENRE_MODEL_DIR=$HOME/hf_models/DeepSeek-R1-Distill-Qwen-7B,WORKERS=1 tests/simuhome/bench/slurm_harness.sbatch
```

Ce am verificat pentru acest raport: am citit codul pasului si al variantelor (fisierele listate in sectiunea 5), am calculat tabelele din
inregistrarile celor 4 rulari (scriptul de mai sus), am citit trace-urile a 6 episoade (4351, 4644, 1085, 2595, 4138, 1303) si a doua cazuri de JSON invalid (1701, 4786),
si am citit `findings.md`, `decisions.md` D55-D61 si `progress.md`. Nu am rulat niciun model si nu am rulat suita de teste (nu am schimbat cod).
Ce nu am facut: nu am verificat articolul GENRE original; nu am masurat zgomotul rularii curat (vezi 6.7); explicatiile marcate [ipoteza] nu sunt testate.

## Anexa: termeni

- **Ontologie / `homeont`**: vocabularul casei (`homeont.ttl`): camere, clase de dispozitive, clase de comenzi; `saref:` e ontologia SAREF pe care se sprijina.
- **Greedy**: la fiecare pas se alege tokenul cel mai probabil. **Beam search**: se tin 4 secvente candidate si se alege cea cu scorul total cel mai bun.
- **Logits**: scorurile brute ale tuturor tokenilor la un pas; **masca** = punerea lui `-inf` pe tokenii nepermisi.
- **`top_logprobs`**: Ollama arata cel mult cei mai probabili 20 de tokeni si probabilitatile lor.
- **Prefill**: trimiterea unui mesaj de asistent neterminat pe care modelul il continua.
- **Trie**: arbore in care un drum de la radacina e un prefix; permite sa intrebi "ce caractere pot urma dupa acest prefix".
- **Gold / parte**: raspunsul asteptat al benchmark-ului, descompus pe comenzi; `error_input` = comanda care trebuie refuzata.
