# Experimentele de structurare pe ECAI-400: rezultate complete, erori, timp (M1-M4, gpt-5, F, G, H, G2, H2, G2-M1, H2-M1)

Data: 2026-11-21 (versiunile precedente: 2026-11-20, fara G2-M1 / H2-M1; 2026-11-17, cu H2 partial). Autor: Coder. Surse: inregistrarile tuturor rularilor din `tests/homebench/results/harness/`, log-urile serverului GENRE
si ale job-urilor de pe fep, `sacct`, codul din repo (commit `c90490eb`; rularea H2 are in `run.json` commitul `22ad219c`, vezi sectiunea 8.1; rularile G2-M1 / H2-M1: `0de6ea3f` / `e15d5ca1`, sectiunea 12). Tabelele sunt generate de
`work/reports/experiments_summary.py` (stdlib, fara LLM; vezi sectiunea 11), nu scrise de mana. Cifrele au eticheta
[masurat] (direct din inregistrari/log-uri), [calculat] (derivat deterministic din ele) sau [ipoteza] (nemasurat).

---

## 0. Citeste asta intai

**Versiunea din 2026-11-21 (a doua): consistenta cu tot ce exista.** Adaugat: sectiunea 13 (performanta si timp, toate rularile complete intr-un tabel, generat) si sectiunea 14 (rularile locale / laptop: ce s-a schimbat fata de `analiza-timp-rulari-locale.md`, ale carui cifre M1 / M3 descriu incercari partiale inlocuite).
Verificat in aceasta sesiune: cod local = `origin` = fep (`e15d5ca1`, cod identic cu cel al ultimelor rulari); toate cele 14 rulari complete de pe fep / laptop sunt copiate local; `error_report` regenerat pe toate cele 14, tabelele rezultate identice cu cele din raport; `structuring_analysis.py` regenerat, cifrele din `structurare_explicat.md` coincid.


**Adaugat 2026-11-21: armele cu Qwen3-8B la structurare, G2-M1 si H2-M1 (detalii in sectiunea 12).**

- **G2-M1** (`iri_trie_v2`, job 303338, `COMPLETED`, 3 h 18 min 26 s) si **H2-M1** (`genre_v2`, patru joburi de cate 100 de episoade, 303360-303363, toate `COMPLETED`, 2 h 38 min - 2 h 59 min fiecare) s-au terminat pe cele 400 de episoade [masurat, `sacct`].
  **G2-M1 = 299 / 392 = 76,3%** (Wilson 95%: 72-80; 8 episoade de infrastructura). **H2-M1 = 297 / 400 = 74,2%** (70-78; 0 episoade de infrastructura). Controlul **M1 = 280 / 385 = 72,7%** (15 de infrastructura).
- **Decodarea constransa nu bate controlul M1 in mod semnificativ.** M1 -> G2-M1: 42 castigate / 28 pierdute, p = 0,12. M1 -> H2-M1: 41 / 36, p = 0,65. G2-M1 -> H2-M1: 27 / 36, p = 0,31 [calculat, McNemar exact].
  Cu DeepSeek-R1-7B aceeasi decodare muta scorul de la 22,6% la 41-44% (p < 0,001); cu Qwen3-8B, care pornea de la 72,7%, castigul de 1,5-3,6 puncte e in zgomot.
- **Zgomotul masurat intre rulari apropiate ale lui M1 e mare**: M1 vs M2 (difera doar planificatorul, care nu conteaza) are 32 castigate / 27 pierdute din 375 de episoade comune (~16% isi schimba verdictul). M1 a rulat pe laptop, G2-M1 / H2-M1 pe fep, cu alt cod,
  deci perechea M1 -> brat nu e un experiment curat (sectiunea 12.5).
- **Ce schimba decodarea la Qwen3 [masurat]**: erorile de notatie dispar (31 de parti la M1, 0 la ambele brate; 29 din ele `struct_parameter_name`) si partile fezabile cresc de la 84,1% la 90,6% (G2-M1) / 88,7% (H2-M1). Partile infezabile nu se misca
  (79,9% -> 79,4% / 81,7%). Decodarea a intervenit rar: 168 (G2-M1) si 174 (H2-M1) de sloturi schimbate, fata de 1932 / 2362 la G2 / H2 cu DeepSeek.
- **Costul H2-M1**: 99,3 s per episod in medie (median 58,5; p95 309; max 977), 11,0 h suma timpilor pe episod; G2-M1: 28,8 s per episod, 3,2 h. Gandirea Qwen3-8B in cererile GENRE are in medie 898 de tokeni (nu ~1,9k cat s-a estimat din 13 apeluri de smoke), taiata si inchisa la plafonul de 10240 in 12 apeluri.

**Raportul de dinainte (2026-11-20), neschimbat:**

1. **H2 (genre_v2) s-a terminat pe cele 400 de episoade.** Job-ul SLURM **302815**: `COMPLETED`, exit 0:0, 5 h 03 min 22 s, 400 inregistrari si 400 `.eval.json` [masurat, `sacct`, folderul de pe fep].
   397 de episoade scorate, 3 de infrastructura (scor -1). **H2 = 164 / 397 = 41,3%** (interval Wilson 95%: 37-46).
2. **H2 e mult mai bun decat H, nu se distinge de G2.** H -> H2: 67 castigate / 30 pierdute, p < 0,001 (32,0% -> 41,3%). G2 -> H2: 46 castigate / 57 pierdute, p = 0,32 (43,8% vs 41,3%):
   diferenta de 2,5 puncte e in zgomot (M3 local vs M3 fep: 8 / 8). G -> H2: 70 / 43, p = 0,014. M3 -> H2: 110 / 37, p < 0,001.
3. **Intrebarea pentru care a fost scris H2 are raspuns: da, repara infezabilele pierdute de H.** hb1 infezabile 32,6% (H) -> 59,3% (H2); partile infezabile 33,5% -> 61,8%. hb1 fezabile 64,9% -> 67,5% (cel mai bun dintre bratele cu decodare).
   Costul: pe cererile cu mai multe parti H2 e sub G2 (multi fezabile 24,5% vs 32,4%; mixed 11,6% vs 16,8%) [masurat; diferenta pe aceste doua grupe nu am testat-o separat].
4. **Reparatia de viteza a functionat.** Prima incercare (job 300721) dura 243,8 s per episod si a cazut la 66 de episoade; dupa `filter_ids` (commit `c90490eb`), pe aceleasi 66 de episoade: 61,4 s (H: 60,7 s). Pe cele 400: beam 6,1 s per apel (H: 5,2 s; incercarea veche: 59,9 s).
   Rularea completa: 4,89 h suma timpilor pe episod, 0 cereri esuate la serverul GENRE, 45,1 GB cgroup la varf (serverul 29,6 GB) [masurat].
5. **Cauza principala a ce a ramas pierdut la H2 este o gaura pe care regulile v2 nu au inchis-o: cheile omise.** `no_null` interzice `null`, iar `keys` limiteaza cheile la cele din raspunsul liber, dar *nimic nu obliga o cheie sa apara*.
   In 110 din 963 de apeluri (11,4%; la H: 64, 6,7%) raspunsul beam nu mai scrie cheia camerei, a clasei dispozitivului sau a clasei actiunii; in 105 din cele 110 cel putin una din cheile lipsa fusese scrisa de raspunsul liber. Cele 90 de episoade atinse trec 12,2%
   (11 / 90), celelalte 49,8% (153 / 307). 20 din cele 22 de episoade `seg_part_lost` ale lui H2 sunt dintre ele [masurat, sectiunea 8.5]. Daca ar trece la rata celorlalte, H2 ar fi la ~50% (estimare-margine, nu masuratoare: episoadele atinse sunt mai grele).
6. **Clasamentul pe cele 400 de episoade** (rata de succes pe episod, episoadele de infrastructura scoase; interval Wilson 95%):
   gpt-5-mini 83,5% > **G2-M1 76,3%** ~ M2 75,0% ~ **H2-M1 74,2%** ~ gpt-5-nano 73,8% ~ M1 72,7% >> **G2 43,8% ~ H2 41,3%** > G 34,3% ~ H 32,0% >> M3 22,6% ~ M4 22,6% ~ M3 local 22,8% > F 19,2%.
7. **Ce conteaza cel mai mult este modelul de structurare, nu decodarea.** Aceeasi cerere cu Qwen3-8B la structurare (M1) trece 72,7%; cu DeepSeek-R1-Distill-Qwen-7B (M3)
   22,6%; cu M3 plus cea mai buna decodare constransa (G2 / H2) 41-44%; cu M1 plus aceeasi decodare (G2-M1 / H2-M1) 74-76%, adica nu se distinge de M1 (sectiunea 12). Planificatorul nu conteaza (M3 -> M4: 90 vs 89 episoade trecute, 8 castigate / 9 pierdute).
8. **Decodarea constransa (G, H, G2, H2) muta eroarea, nu o elimina.** Erorile de notatie (`struct_class_prefix`, `slot_confusion`, `invalid_device_name`: 403 de parti la M3) dispar
   (G 0, G2 1, H 7, H2 0), dar apar altele: valori gresite, actiune gresita, dispozitiv substituit, parti pierdute intre structurare si plan (sectiunea 6).
9. **Diferentele mici sunt zgomot.** Aceeasi configuratie rulata pe doua masini (M3 local vs M3 fep): 90 vs 90 episoade trecute,
   dar 8 castigate si 8 pierdute (4% din episoade isi schimba verdictul) [masurat]. G -> H: 52 / 63, p = 0,35; G2 -> H2: p = 0,32: nu se poate spune care din fiecare pereche e mai bun.
   M3 -> G, M3 -> H, M3 -> G2, M3 -> H2, G -> G2, H -> G2, H -> H2 sunt diferente reale (p <= 0,003).
   Pe Qwen3-8B, M1 vs M2 (aceeasi structurare, alt planificator): 32 / 27, p = 0,60, deci ~16% din episoade isi schimba verdictul fara nicio diferenta relevanta; M1 -> G2-M1 (42 / 28, p = 0,12) si M1 -> H2-M1 (41 / 36, p = 0,65) sunt in aceasta banda.

---

## 1. Cum se citesc numerele

- **Episod trecut** = toate partile gold sunt indeplinite (scorul principal al harness-ului). **Parte** = un token gold (o comanda sau un `error_input` care trebuie refuzat).
- **Infra** = episod pe care harness-ul l-a scorat -1 (API, simulator sau judecator esuat, nu agentii). E scos din orice rata si din orice comparatie pereche.
  Nu e acelasi lucru cu *stadiul* `infrastructure` al `error_report` (primul stadiu care a pierdut o parte), care e mai larg: de aceea la unele rulari exista
  episoade "fail" cu cod `infra_*` (M1: 11, H: 15, H2: 0, G2-M1: 13, H2-M1: 0) pe langa cele "infra".
- **Podeaua**: un agent care nu face nimic trece 100% din cererile infezabile si ~10,7% din cele fezabile (known-issues). M3 are 88-92% pe infezabile pentru ca
  refuza aproape orice (structuri gresite -> planificatorul zice "imposibil"), nu pentru ca intelege. De aceea tabelul de mai jos da si **partile fezabile** si
  **partile infezabile** separat: un brat bun le tine pe amandoua sus.
- **Interval 95%**: Wilson, pe episoadele scorate. Cu n ~ 400 e de aproximativ +/- 5 puncte; doua rulari ale caror intervale se suprapun nu se pot ordona fara testul pe perechi.
- **Test pe perechi**: acelasi episod in doua rulari; "castigate" = A pica, B trece; "pierdute" invers; **McNemar exact** pe discordante (p bilateral).
- **Zgomot**: vezi M3 local -> M3 fep din sectiunea 3.1 (acelasi cod si model, alta masina): ~16 din 398 de episoade isi schimba verdictul.
- Rularile **nu sunt toate pe acelasi cod**: comiturile difera (coloana `commit`), scorul rularilor vechi (30 oct - 1 nov) a fost scris in `.eval.json` la acea data. Episoadele sunt aceleasi 400 (ECAI-400, cu cele `_fixed`).

---

## 2. Catalogul experimentelor: ce are diferit fiecare

Toate rularile folosesc aceleasi 400 de episoade, `environment: on`, memorie oprita, garda oprita (HomeBench). Ce difera e **doar** ce e scris mai jos
(plus infrastructura: coloana `workers`, nodul, commitul). Linia `H2p` din tabel e prima incercare de H2 (job 300721, cazuta la 66 de episoade, pastrata doar pentru timpi).

#### Runs

| key | folder | structuring model | decoding | workers | commit | started | records | scored | infra eps | cluster job |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt5m | `homebench.ecai.gpt-5-mini` | gpt-5-mini | free | 4 | `68e51755` | 2026-10-30 18:12 | 400 | 400 | 0 | - |
| gpt5n | `homebench.ecai.gpt-5-nano` | gpt-5-nano | free | 4 | `68e51755` | 2026-10-30 16:50 | 400 | 400 | 0 | - |
| M1 | `homebench.ecai.M1` | qwen3-8b-16k | free | 1 | `68e51755` | 2026-10-31 13:10 | 400 | 385 | 15 | - |
| M2 | `homebench.ecai.M2` | qwen3-8b-16k | free | 1 | `68e51755` | 2026-10-31 20:09 | 400 | 388 | 12 | - |
| M4 | `homebench.ecai.M4` | ds-r1-7b-16k | free | 1 | `68e51755` | 2026-11-01 08:50 | 400 | 399 | 1 | - |
| M3L | `homebench.ecai.M3` | ds-r1-7b-16k | free | 1 | `68e51755` | 2026-11-01 04:46 | 400 | 399 | 1 | - |
| F | `homebench.ecai.M3.pv-json-v2` | ds-r1-7b-16k | free + context json_v2 | 1 | `00559259` | 2026-11-09 12:05 | 400 | 396 | 4 | - |
| M3 | `fep400.M3` | ds-r1-7b-16k | free | 4 | `bfb099d4` | 2026-11-07 12:18 | 400 | 399 | 1 | 294585 1:06:48 COMPLETED |
| G | `fep.trie-v1` | ds-r1-7b-16k | iri_trie | 4 | `bf3be605` | 2026-11-15 13:13 | 400 | 394 | 6 | 299621 1:56:15 COMPLETED |
| H | `fep.genre-v1` | DeepSeek-R1-Distill-Qwen-7B | genre | 1 | `bf3be605` | 2026-11-16 17:04 | 400 | 397 | 3 | 300297 4:45:30 COMPLETED |
| G2 | `fep.trie-v2` | ds-r1-7b-16k | iri_trie_v2 | 1 | `22ad219c` | 2026-11-16 23:13 | 400 | 393 | 7 | 300435 2:10:12 COMPLETED |
| H2 | `fep.genre-v2` | DeepSeek-R1-Distill-Qwen-7B | genre_v2 | 1 | `22ad219c` | 2026-11-19 23:22 | 400 | 397 | 3 | 302815 5:03:22 COMPLETED |
| G2M1 | `fep.trie-v2-m1` | qwen3-8b-16k | iri_trie_v2 | 1 | `0de6ea3f` | 2026-11-20 16:26 | 400 | 392 | 8 | 303338 3:18:26 COMPLETED |
| H2M1 | `fep.genre-v2-m1` | Qwen3-8B | genre_v2 | 1 | `e15d5ca1` | 2026-11-20 16:59 | 400 | 400 | 0 | 303360-303363 2:37:42 + 2:58:32 + 2:47:10 + 2:49:03 COMPLETED |
| H2p | `fep.genre-v2-partial` | DeepSeek-R1-Distill-Qwen-7B | genre_v2 | 1 | `22ad219c` | 2026-11-17 10:35 | 66 | 66 | 0 | 300721 4:33:08 FAILED (exit 120) |


| brat | ce schimba fata de M3 | model / backend | de ce exista |
|---|---|---|---|
| **M3** | nimic (control) | segmentare Qwen3-8B, **structurare DS-R1-Distill-Qwen-7B** (Ollama, `ds-r1-7b-16k`), planificare + raspuns Qwen2.5-Coder-7B, text liber, JSON parsat lax | referinta pentru toate bratele de structurare |
| **M3L** (M3 local) | acelasi config ca M3, rulat pe masina locala (Ollama pe localhost, 1 worker, 1 nov) | idem | masoara zgomotul intre doua rulari "identice" |
| **M4** | planificator si raspuns Qwen3-4B in loc de Qwen2.5-Coder-7B | idem | arata ca planificatorul nu conteaza (explicit = SPARQL deterministic) |
| **M1** | **structurare Qwen3-8B** (profilare Qwen3-8B, neapelata pe HomeBench) | idem | cat se castiga doar schimband modelul de structurare |
| **M2** | M1 cu planificator Qwen3-4B | idem | idem, cu planificator mic |
| **gpt-5-mini / gpt-5-nano** | toti pasii cu modelul OpenAI | idem | plafon cu un model de frontiera mic, 4 workers |
| **F** (`pv-json-v2`) | `context: json_v2`: contextul casei trimis ca JSON cu regulile rescrise in chei, in loc de text indentat | acelasi model, alt format al promptului | testeaza daca formatul contextului repara notatia |
| **G** (`pv-trie`, `iri_trie`) | dupa raspunsul liber, fiecare slot IRI (camera, clasa dispozitivului, clasa si numele actiunii, ...) e redecodat un token pe rand, cu un **trie de IRI-uri valide ale casei**, conditionat de sloturile deja scrise | Ollama, `top_logprobs` 20 + prefill; ~16 apeluri de structurare/episod; fallback `nearest` | elimina notatia gresita prin constructie |
| **H** (`pv-genre`, `genre`) | **GENRE real**: modelul ruleaza in-process (Hugging Face, ponderi bf16), gandire libera, raspuns liber, apoi **beam search (4 fascicule) sub o masca de logits** pe tot vocabularul (trie de tokeni + masca de validitate JSON) | un apel de structurare per scop, cu trei generari; alt backend decat M3 (ponderi si template diferite) => **M3 nu e control curat pentru H**; fara abstinenta | GENRE fidel, un singur apel de structurare |
| **G2** (`pv-trie-v2`, `iri_trie_v2`) | G + **regula evidentei** (sloturile scrise mai tarziu restrang trie-ul celor scrise mai devreme) + **abstinenta** (un dispozitiv real absent din camera numita devine `unmatched_device`, nu e inlocuit cu cel mai apropiat) | Ollama, ~12 apeluri/episod | repara 4351 (clasa ghicita) si substitutiile infezabile ale lui G |
| **G2-M1** (`M1.pv-trie-v2`, `iri_trie_v2`) | exact G2, dar structurarea e facuta de **Qwen3-8B** (`qwen3-8b-16k`, Ollama pe nodul jobului), nu de DeepSeek; controlul e **M1** | Ollama, cerere de 180 s ca la M1; ~3,0 apeluri de structurare / episod | arata daca decodarea ajuta si un model care scrie deja notatia corecta |
| **H2-M1** (`M1.pv-genre-v2`, `genre_v2`) | exact H2, dar structurare **Qwen3-8B** (`Qwen3-8B`, Hugging Face bf16, serverul `genre_server.py`, `GENRE_MAX_TOKENS=10240`, `BENCH_REQUEST_TIMEOUT=1500`); 4 joburi x 100 de episoade, rezultatele unite | restul pasilor pe Ollama ca la M1; **alt backend la structurare decat controlul M1** | idem pentru GENRE |
| **H2** (`pv-genre-v2`, `genre_v2`) | H + `no_null` (fara `null` pentru camera/dispozitiv/actiune cat timp casa are valori) + `keys` (cheile JSON raman cele ale raspunsului liber) + `close_thinking` (gandire taiata la plafon e inchisa cu `</think>`) + abstinenta citita din raspunsul liber; masca de tokeni luata din trie (`filter_ids`, `c90490eb`) | acelasi server HF ca H; **terminat, 400 de episoade** | repara cele 3 cauze ale pierderilor lui H pe infezabile |

Brate care exista in cod dar **fara rezultate** (nu au folder in `results/harness/`): **B** `pv-schema` (`response_format` json_schema strict), **C** `pv-json` (JSON v1),
**D** `pv-focused` (prompt scurtat cu 2 exemple). Bratul **E** `pv-repair` (`repair: class_names`) a rulat doar pe 20 de episoade (sectiunea 4.3). Rularile partiale
(M5, qwen3-8b singur, qwen2.5-coder-7b singur) nu au inregistrari scorate si nu intra in comparatii; timpii lor sunt in `analiza-timp-rulari-locale.md`.

---

## 3. Rezultate pe cele 400 de episoade

#### Episode pass rate (infra left out)

| run | scored | pass | 95% CI | hb1 feasible | hb1 infeasible | multi feasible | mixed | feasible parts | infeasible parts | EM | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt5m | 400 | 334 = 83.5% | 80-87 | 113/114 = 99.1% | 72/86 = 83.7% | 91/104 = 87.5% | 58/96 = 60.4% | 96.0% | 79.8% | 83 | 92 | 92 | 92 |
| gpt5n | 400 | 295 = 73.8% | 69-78 | 107/114 = 93.9% | 65/86 = 75.6% | 70/104 = 67.3% | 53/96 = 55.2% | 89.0% | 77.4% | 72 | 92 | 85 | 88 |
| M1 | 385 | 280 = 72.7% | 68-77 | 98/109 = 89.9% | 69/81 = 85.2% | 68/102 = 66.7% | 45/93 = 48.4% | 84.1% | 79.9% | 72 | 87 | 82 | 85 |
| M2 | 388 | 291 = 75.0% | 70-79 | 106/114 = 93.0% | 72/81 = 88.9% | 69/102 = 67.6% | 44/91 = 48.4% | 84.7% | 82.8% | 75 | 88 | 84 | 86 |
| M4 | 399 | 90 = 22.6% | 19-27 | 8/114 = 7.0% | 79/86 = 91.9% | 0/103 = 0.0% | 3/96 = 3.1% | 11.1% | 97.3% | 20 | 29 | 27 | 28 |
| M3L | 399 | 91 = 22.8% | 19-27 | 8/114 = 7.0% | 79/86 = 91.9% | 0/103 = 0.0% | 4/96 = 4.2% | 11.4% | 97.3% | 20 | 29 | 27 | 28 |
| F | 396 | 76 = 19.2% | 16-23 | 10/113 = 8.8% | 63/85 = 74.1% | 0/103 = 0.0% | 3/95 = 3.2% | 11.0% | 89.7% | 16 | 29 | 25 | 26 |
| M3 | 399 | 90 = 22.6% | 19-27 | 9/114 = 7.9% | 75/85 = 88.2% | 0/104 = 0.0% | 6/96 = 6.2% | 12.4% | 95.7% | 20 | 29 | 27 | 28 |
| G | 394 | 135 = 34.3% | 30-39 | 58/113 = 51.3% | 41/84 = 48.8% | 28/102 = 27.5% | 8/95 = 8.4% | 55.3% | 57.2% | 32 | 62 | 53 | 57 |
| H | 397 | 127 = 32.0% | 28-37 | 74/114 = 64.9% | 28/86 = 32.6% | 21/102 = 20.6% | 4/95 = 4.2% | 59.4% | 33.5% | 30 | 70 | 51 | 59 |
| G2 | 393 | 172 = 43.8% | 39-49 | 74/112 = 66.1% | 49/84 = 58.3% | 33/102 = 32.4% | 16/95 = 16.8% | 64.3% | 64.7% | 42 | 69 | 62 | 66 |
| H2 | 397 | 164 = 41.3% | 37-46 | 77/114 = 67.5% | 51/86 = 59.3% | 25/102 = 24.5% | 11/95 = 11.6% | 62.0% | 61.8% | 40 | 72 | 60 | 66 |
| G2M1 | 392 | 299 = 76.3% | 72-80 | 102/111 = 91.9% | 70/83 = 84.3% | 79/103 = 76.7% | 48/95 = 50.5% | 90.6% | 79.4% | 76 | 90 | 88 | 89 |
| H2M1 | 400 | 297 = 74.2% | 70-78 | 103/114 = 90.4% | 75/86 = 87.2% | 74/104 = 71.2% | 45/96 = 46.9% | 88.7% | 81.7% | 74 | 90 | 87 | 88 |


Rezumat pe pasii logici:

| pas | succes | hb1 fezabile | hb1 infezabile | parti fezabile | parti infezabile |
|---|---|---|---|---|---|
| M3 (control) | 22,6% | 7,9% | 88,2% | 12,4% | 95,7% |
| G: trie | 34,3% | 51,3% | 48,8% | 55,3% | 57,2% |
| H: GENRE | 32,0% | 64,9% | 32,6% | 59,4% | 33,5% |
| **G2**: trie + evidenta + abstinenta | **43,8%** | 66,1% | 58,3% | 64,3% | 64,7% |
| **H2**: GENRE + no_null + keys + abstinenta | 41,3% | **67,5%** | **59,3%** | 62,0% | 61,8% |
| M1: alt model de structurare, fara decodare | 72,7% | 89,9% | 85,2% | 84,1% | 79,9% |
| **G2-M1**: M1 + trie + evidenta + abstinenta | 76,3% | 91,9% | 84,3% | 90,6% | 79,4% |
| **H2-M1**: M1 + GENRE v2 | 74,2% | 90,4% | 87,2% | 88,7% | 81,7% |
| gpt-5-mini | 83,5% | 99,1% | 83,7% | 96,0% | 79,8% |

Citire: decodarea constransa a mutat partile fezabile de la 12% la 55-64%, dar a **platit pe infezabile** (96% -> 33-65%), pentru ca M3 trecea infezabilele prin refuz
generalizat. G2 si H2 sunt singurele brate cu decodare care au ambele jumatati la ~62-65%. H2 are cele mai bune hb1 (ambele), G2 cel mai bun total, prin grupele cu mai multe parti.
Dar M1, doar cu alt model de structurare, are 84% / 80% fara nicio decodare speciala. Cu Qwen3-8B decodarea (G2-M1, H2-M1) ridica partile fezabile la 89-91% si lasa infezabilele la 79-82%:
la acest model nu mai are de reparat o notatie gresita, iar greselile ramase (valoare, dispozitiv infezabil "reparat" cu altul, clarificari false) nu sunt de notatie (sectiunea 12.3).

### 3.1 Perechi (acelasi episod, doua rulari)

#### Paired flips (same episode, two runs; infra left out)

| pair | n | gained | lost | pass A | pass B | McNemar p |
|---|---|---|---|---|---|---|
| M3 -> G | 393 | 87 | 40 | 88 | 135 | <0.001 |
| M3 -> H | 396 | 94 | 57 | 90 | 127 | 0.003 |
| M3 -> G2 | 393 | 116 | 33 | 89 | 172 | <0.001 |
| M3 -> H2 | 396 | 110 | 37 | 90 | 163 | <0.001 |
| G -> G2 | 388 | 64 | 30 | 135 | 169 | <0.001 |
| H -> H2 | 397 | 67 | 30 | 127 | 164 | <0.001 |
| G -> H | 392 | 52 | 63 | 135 | 124 | 0.351 |
| G2 -> H2 | 393 | 46 | 57 | 172 | 161 | 0.324 |
| G -> H2 | 392 | 70 | 43 | 135 | 162 | 0.014 |
| M3L -> M3 | 398 | 8 | 8 | 90 | 90 | 1.000 |
| M3L -> F | 396 | 7 | 21 | 90 | 76 | 0.013 |
| M3 -> F | 395 | 7 | 21 | 89 | 75 | 0.013 |
| M1 -> M2 | 375 | 32 | 27 | 274 | 279 | 0.603 |
| M3 -> M4 | 398 | 8 | 9 | 90 | 89 | 1.000 |
| M1 -> M3L | 385 | 12 | 207 | 280 | 85 | <0.001 |
| gpt5m -> M1 | 385 | 12 | 54 | 322 | 280 | <0.001 |
| gpt5m -> G2 | 393 | 14 | 170 | 328 | 172 | <0.001 |
| gpt5m -> H2 | 397 | 11 | 179 | 332 | 164 | <0.001 |
| M1 -> G2M1 | 378 | 42 | 28 | 274 | 288 | 0.120 |
| M1 -> H2M1 | 385 | 41 | 36 | 280 | 285 | 0.649 |
| G2M1 -> H2M1 | 392 | 27 | 36 | 299 | 290 | 0.314 |
| G2 -> G2M1 | 387 | 141 | 12 | 167 | 296 | <0.001 |
| H2 -> H2M1 | 397 | 147 | 14 | 164 | 297 | <0.001 |
| gpt5m -> G2M1 | 392 | 14 | 43 | 328 | 299 | <0.001 |
| gpt5m -> H2M1 | 400 | 20 | 57 | 334 | 297 | <0.001 |


Ce spun: (a) **M3L -> M3 = 8 / 8** (acelasi config, alta masina) este zgomotul: ~16 episoade pe 398. (b) **M1 -> M3L = 12 castigate / 207 pierdute**: modelul de structurare, nu decodarea,
explica cea mai mare parte din diferenta. (c) **M1 -> M2 (p = 0,60)** si **M3 -> M4 (p = 1,0)**: planificatorul nu conteaza. (d) **F** (JSON) e *mai rau* decat M3 (7 castigate / 21 pierdute, p = 0,013).
(e) **G -> H nu se distinge** (p = 0,35); **H -> G2, G -> G2 si H -> H2 sunt reale** (p < 0,001). (f) **G2 -> H2 nu se distinge** (46 / 57, p = 0,32); G -> H2 e real (p = 0,014).
(g) Fiecare brat din perechea (G, H) -> (G2, H2) a castigat ~65 de episoade si a pierdut ~30 (G -> G2: 64 / 30; H -> H2: 67 / 30): regulile v2 au efect aproape identic pe cele doua mecanisme.

---

## 4. Pe tipuri de cereri

Grupurile sunt cele ale `error_report` (bucket): **hb1 fezabile** (o comanda), **hb1 infezabile** (o comanda de refuzat), **multi fezabile** (2-10 comenzi), **mixed** (fezabile + de refuzat).
Coloanele de grup din tabelul din sectiunea 3 sunt aceste grupuri.

- **Multi fezabile**: M3 0/104; G 27,5%; H 20,6%; G2 32,4%; H2 24,5%; M1 66,7%; gpt-5-mini 87,5%. O cerere cu N parti trece doar daca toate cele N sunt corecte, deci erorile se inmultesc:
  chiar la 64% pe parte, un episod cu 3 parti trece ~26% din ori.
- **Mixed**: M3 6,2%; G 8,4%; H 4,2%; G2 16,8%; H2 11,6%; M1 48,4%; gpt-5-mini 60,4%. Cea mai grea categorie pentru toate bratele de decodare. 44 din cele 90 de episoade ale lui H2 cu o cheie omisa
  (sectiunea 8.5) sunt mixed: aici se vede cel mai mult gaura, iar un episod cu mai multe parti pica daca un singur apel de structurare pierde o cheie.
- **Cererile relative** ("increase ... by 34%", "lower ... by 5 degrees") sunt cea mai mare pierdere ramasa la hb1 fezabile, indiferent de decodare: modelul scrie `goal_effect: set`
  cu valoarea cererii (o diferenta), iar gold-ul e absolut. Detaliu (rulari cu decodare):

#### hb1 feasible, relative requests: `goal_effect` written and pass

| run | request | goal_effect | n | pass |
|---|---|---|---|---|
| M3 | relative | modify | 16 | 0 (0.0%) |
| M3 | relative | set | 26 | 0 (0.0%) |
| M3 | absolute | modify | 1 | 0 (0.0%) |
| M3 | absolute | set | 69 | 8 (11.6%) |
| M3 | absolute | None | 1 | 1 (100.0%) |
| G | relative | modify | 19 | 11 (57.9%) |
| G | relative | set | 24 | 0 (0.0%) |
| G | absolute | modify | 6 | 0 (0.0%) |
| G | absolute | set | 64 | 47 (73.4%) |
| H | relative | modify | 20 | 13 (65.0%) |
| H | relative | set | 20 | 1 (5.0%) |
| H | relative | None | 2 | 0 (0.0%) |
| H | absolute | modify | 6 | 1 (16.7%) |
| H | absolute | set | 64 | 58 (90.6%) |
| G2 | relative | modify | 22 | 15 (68.2%) |
| G2 | relative | set | 20 | 0 (0.0%) |
| G2 | absolute | modify | 5 | 1 (20.0%) |
| G2 | absolute | set | 65 | 58 (89.2%) |
| H2 | relative | modify | 23 | 18 (78.3%) |
| H2 | relative | set | 19 | 1 (5.3%) |
| H2 | relative | None | 1 | 0 (0.0%) |
| H2 | absolute | modify | 6 | 2 (33.3%) |
| H2 | absolute | set | 65 | 56 (86.2%) |
| M1 | relative | modify | 35 | 33 (94.3%) |
| M1 | relative | set | 3 | 0 (0.0%) |
| M1 | absolute | modify | 1 | 1 (100.0%) |
| M1 | absolute | set | 70 | 64 (91.4%) |
| G2M1 | relative | modify | 37 | 34 (91.9%) |
| G2M1 | relative | set | 3 | 0 (0.0%) |
| G2M1 | absolute | modify | 1 | 1 (100.0%) |
| G2M1 | absolute | set | 70 | 67 (95.7%) |
| H2M1 | relative | modify | 40 | 37 (92.5%) |
| H2M1 | relative | set | 1 | 0 (0.0%) |
| H2M1 | relative | None | 2 | 0 (0.0%) |
| H2M1 | absolute | set | 70 | 66 (94.3%) |
| H2M1 | absolute | None | 1 | 0 (0.0%) |


  `set` pentru o cerere relativa: 0 din 24 trec la G, 1 din 20 la H, 0 din 20 la G2, 1 din 19 la H2. Cu `modify` trec 58-78% (H2: 18 din 23). Nu e o problema de decodare, ci una de regula
  (propusa in `structurare_explicat.md` 7.1, nemasurata ca brat).

### 4.1 Cererile infezabile: costul decodarii

La M3 "92% pe infezabile" e podeaua. La G / H / G2 / H2 partile infezabile trecute scad la 57% / 34% / 65% / 62%. Codurile: `infeas_substituted` (un alt dispozitiv din camera a fost operat in loc
sa se raporteze lipsa) 33 la G, 16 la H, 17 la G2, 21 la H2; `infeas_executed_other` 10 / 14 / 21 / **6**; `struct_false_clarify_device` ("nu gasesc X" pentru un dispozitiv care exista) 2 / 4 / 14 / 12.
G2 a schimbat un tip de eroare cu altul: abstinenta a oprit o parte din substitutii, dar regula evidentei a inlocuit camera corecta in 36 de episoade (`structurare_explicat.md` 7.2).
H2 are abstinenta fara regula evidentei (69 de abstineri: 50 in mixed, 16 in infezabile, 3 in fezabile, deci cel mult 3 fals pozitive) si cel mai putin `infeas_executed_other`.

### 4.2 Etichete suspecte

`infeas_accepted_on_target` (casa a acceptat actiunea pe dispozitivul "imposibil"): 12 / 11 / 10 / 14 parti la G / H / G2 / H2 (gpt-5-mini: 1, gpt-5-nano: 8). 657 de randuri
`unexist_attribute` contrazic casa (documentul pentru Alex). Un brat care *executa* corect aceste cereri e penalizat de eticheta.

### 4.3 Pilotii pe 20 de episoade (14 nov, hb1, masina locala)

Inainte de rularile pe fep, s-au facut piloti pe 20 de episoade hb1 (12 fezabile, 8 infezabile; nu am re-derivat cum au fost alesi): control 11/20 (55%), E `repair: class_names`
13/20 (65%), G `pv-trie.v1` 12/20 (60%), G `pv-trie` (a doua rulare) 12/20 (60%) (`comparison_tables.md`). n = 20 nu permite nicio concluzie statistica; au servit la depanare (bugul 4351), nu la comparatie.

---

## 5. Tipuri de erori

Un episod pica la **primul stadiu** care a pierdut una din partile lui. Cum se citesc codurile: glosarul din `tests/homebench/error_taxonomy.py` (`CODES`); cele frecvente:
`struct_class_prefix` clasa scrisa fara prefixul `homeont:` / `saref:`; `struct_slot_confusion` valoare in campul gresit; `struct_invalid_device_name` nume de dispozitiv inventat;
`struct_wrong_value` valoarea tinta difera de gold; `struct_room_missing` scop fara camera; `seg_part_lost` nicio structura pentru o parte; `plan_unrouted` planificatorul nu accepta structura;
`plan_false_impossible` planificatorul zice "imposibil" pentru o parte fezabila; `exec_wrong_value` comanda corecta cu alta valoare; `infeas_substituted` s-a operat alt dispozitiv;
`infeas_executed_other` nimic refuzat, actiuni executate in alta parte; `eval_turn_off_reading` trecut doar prin citirea permisiva "turn off = luminozitate 0".

### 5.1 Stadiul care pierde episodul

#### Failing episodes by the first stage that lost them

| stage | gpt5m | gpt5n | M1 | M2 | M4 | M3L | F | M3 | G | H | G2 | H2 | G2M1 | H2M1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| evaluation | 6 | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 2 | 6 | 4 |
| execution | 2 | 6 | 9 | 7 | 0 | 0 | 0 | 0 | 23 | 19 | 23 | 21 | 11 | 10 |
| feasibility | 28 | 28 | 23 | 19 | 0 | 0 | 0 | 1 | 32 | 21 | 34 | 28 | 25 | 22 |
| infrastructure | 0 | 0 | 11 | 9 | 1 | 1 | 5 | 2 | 3 | 15 | 1 | 0 | 13 | 0 |
| planning | 1 | 11 | 1 | 3 | 6 | 6 | 30 | 8 | 42 | 41 | 29 | 30 | 1 | 9 |
| segmentation | 5 | 4 | 3 | 3 | 2 | 2 | 6 | 2 | 7 | 48 | 9 | 22 | 3 | 3 |
| structuring | 24 | 52 | 54 | 52 | 300 | 299 | 279 | 296 | 152 | 124 | 124 | 130 | 34 | 55 |
| infra (verdict) | 0 | 0 | 15 | 12 | 1 | 1 | 4 | 1 | 6 | 3 | 7 | 3 | 8 | 0 |
| failing episodes | 66 | 105 | 105 | 97 | 309 | 308 | 320 | 309 | 259 | 270 | 221 | 233 | 93 | 103 |


### 5.2 Familii de erori (parti picate)

#### Failing parts by family of error

| family | gpt5m | gpt5n | M1 | M2 | M4 | M3L | F | M3 | G | H | G2 | H2 | G2M1 | H2M1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| notation: right idea, wrong spelling | 0 | 30 | 31 | 32 | 416 | 409 | 271 | 403 | 0 | 7 | 1 | 0 | 0 | 0 |
| structuring chose wrong (room, device, action, value) | 1 | 20 | 23 | 21 | 225 | 227 | 320 | 226 | 217 | 184 | 157 | 174 | 16 | 25 |
| structuring reply unusable | 0 | 2 | 2 | 0 | 0 | 0 | 1 | 2 | 6 | 0 | 6 | 0 | 0 | 0 |
| lost between structuring and plan | 11 | 26 | 8 | 12 | 13 | 13 | 67 | 15 | 82 | 143 | 64 | 84 | 7 | 17 |
| executed with a wrong value | 3 | 7 | 13 | 10 | 1 | 2 | 4 | 4 | 59 | 54 | 47 | 50 | 12 | 11 |
| infeasible part mishandled | 55 | 38 | 37 | 39 | 1 | 3 | 2 | 2 | 47 | 40 | 56 | 44 | 60 | 61 |
| label suspect (home accepted the action) | 1 | 8 | 2 | 1 | 0 | 0 | 0 | 0 | 12 | 11 | 10 | 14 | 2 | 3 |
| scoring reading (turn off as brightness 0) | 10 | 7 | 10 | 10 | 0 | 0 | 0 | 0 | 3 | 7 | 4 | 5 | 10 | 10 |
| infrastructure (timeout, LLM error, context) | 0 | 0 | 37 | 25 | 1 | 1 | 6 | 2 | 3 | 16 | 1 | 0 | 13 | 0 |
| other codes | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |


Familia "pierdute intre structurare si plan" (`seg_part_lost`, `plan_unrouted`, ...): G 82, H 143, G2 64, **H2 84**. H -> H2 taie aproape jumatate (143 -> 84), dar H2 ramane peste G2 (64):
diferenta e `seg_part_lost` 30 vs 14 si `struct_room_missing` 29 vs 7, adica exact efectul cheilor omise (sectiunea 8.5).

### 5.3 Toate codurile

#### Failing parts by error code

| code | gpt5m | gpt5n | M1 | M2 | M4 | M3L | F | M3 | G | H | G2 | H2 | G2M1 | H2M1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `struct_wrong_value` | 0 | 12 | 21 | 20 | 96 | 90 | 32 | 89 | 119 | 79 | 126 | 108 | 14 | 19 |
| `struct_class_prefix` | 0 | 0 | 1 | 1 | 256 | 253 | 43 | 234 | 0 | 0 | 0 | 0 | 0 | 0 |
| `struct_wrong_room` | 0 | 2 | 1 | 0 | 78 | 82 | 239 | 82 | 6 | 1 | 1 | 1 | 1 | 2 |
| `struct_slot_confusion` | 0 | 0 | 0 | 0 | 118 | 113 | 102 | 120 | 0 | 0 | 1 | 0 | 0 | 0 |
| `exec_wrong_value` | 3 | 7 | 13 | 10 | 1 | 2 | 4 | 4 | 59 | 54 | 47 | 50 | 12 | 11 |
| `infeas_substituted` | 28 | 27 | 21 | 22 | 0 | 0 | 0 | 1 | 33 | 16 | 17 | 21 | 27 | 24 |
| `struct_invalid_device_name` | 0 | 0 | 0 | 0 | 27 | 28 | 124 | 38 | 0 | 0 | 0 | 0 | 0 | 0 |
| `struct_wrong_affordance` | 1 | 2 | 1 | 1 | 17 | 17 | 7 | 18 | 64 | 22 | 12 | 23 | 1 | 1 |
| `struct_room_missing` | 0 | 1 | 0 | 0 | 17 | 21 | 16 | 11 | 10 | 70 | 7 | 29 | 0 | 0 |
| `seg_part_lost` | 9 | 7 | 5 | 6 | 2 | 2 | 6 | 2 | 9 | 62 | 14 | 30 | 6 | 9 |
| `plan_unrouted` | 0 | 7 | 0 | 0 | 5 | 6 | 23 | 8 | 23 | 41 | 11 | 11 | 0 | 7 |
| `plan_explicit_failed` | 1 | 3 | 1 | 3 | 4 | 2 | 3 | 1 | 30 | 25 | 18 | 26 | 1 | 1 |
| `struct_parameter_name` | 0 | 0 | 29 | 30 | 12 | 12 | 2 | 7 | 0 | 1 | 0 | 0 | 0 | 0 |
| `infra_llm_timeout` | 0 | 0 | 37 | 25 | 1 | 1 | 2 | 2 | 3 | 0 | 1 | 0 | 13 | 0 |
| `struct_false_clarify_attribute` | 14 | 3 | 10 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 29 |
| `struct_false_clarify_device` | 9 | 4 | 6 | 2 | 1 | 3 | 2 | 1 | 2 | 4 | 14 | 12 | 14 | 8 |
| `struct_value_missing` | 0 | 1 | 0 | 0 | 11 | 11 | 10 | 20 | 8 | 8 | 7 | 5 | 0 | 0 |
| `eval_turn_off_reading` | 10 | 7 | 10 | 10 | 0 | 0 | 0 | 0 | 3 | 7 | 4 | 5 | 10 | 10 |
| `plan_false_impossible` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 15 | 21 | 16 | 0 | 0 |
| `struct_wrong_device_class` | 0 | 2 | 0 | 0 | 6 | 6 | 16 | 6 | 10 | 4 | 4 | 8 | 0 | 3 |
| `infeas_accepted_on_target` | 1 | 8 | 2 | 1 | 0 | 0 | 0 | 0 | 12 | 11 | 10 | 14 | 2 | 3 |
| `plan_no_action` | 1 | 6 | 2 | 3 | 2 | 3 | 35 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| `infeas_executed_other` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 14 | 21 | 6 | 0 | 0 |
| `struct_device_unnamed` | 0 | 30 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| `struct_json_invalid` | 0 | 2 | 2 | 0 | 0 | 0 | 1 | 2 | 6 | 0 | 6 | 0 | 0 | 0 |
| `infeas_silent` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 6 | 4 | 5 | 1 | 0 |
| `infra_llm_error` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 0 |
| `struct_false_clarify_other` | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| `struct_wrong_device_name` | 0 | 0 | 0 | 0 | 2 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `struct_placeholder_echo` | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `infra_llm_context` | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `struct_false_clarify_ambiguous` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `plan_missing_entry` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `plan_error_other` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `plan_false_clarify` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| `plan_unplannable` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| all failing parts | 81 | 139 | 163 | 150 | 657 | 655 | 671 | 654 | 429 | 462 | 346 | 371 | 120 | 130 |
| parts in scored episodes | 991 | 991 | 962 | 950 | 988 | 988 | 977 | 990 | 970 | 976 | 972 | 976 | 979 | 991 |


Cum s-a mutat eroarea de la H la H2 (parti picate): `struct_room_missing` 70 -> 29, `seg_part_lost` 62 -> 30, `plan_unrouted` 41 -> 11, `infra_llm_error` 16 -> 0, `infeas_executed_other` 14 -> 6, toate partile picate 462 -> 371.
In sens invers: `struct_wrong_value` 79 -> 108 (partile salvate din primele stadii ajung la valoare si pica acolo), `infeas_substituted` 16 -> 21, `struct_false_clarify_device` 4 -> 12 (abstinenta are fals pozitive).
`struct_wrong_value` (108) ramane cea mai mare categorie la H2, ca la G2 (126): decodarea nu atinge valoarea tinta.

---

## 6. Ce a facut fiecare decodare, in cifre

#### What the decodings did (structuring calls with a decoding record)

- **G**: 955 structuring calls with a record; 6386 model calls; changed/decoded slots 2329 ({'location_class': 868, 'parameter_name': 538, 'affordance_class': 332, 'affordance_name': 235, 'artifact_name': 191, 'artifact_class': 162, 'space_class': 1}); how {'trie': 2273, 'nearest': 56}; abstentions by case {}
- **G2**: 957 structuring calls with a record; 4871 model calls; changed/decoded slots 1932 ({'location_class': 860, 'parameter_name': 485, 'affordance_class': 279, 'artifact_class': 119, 'artifact_name': 113, 'affordance_name': 68, 'space_name': 5}); how {'trie': 1846, 'nearest': 18, 'abstain': 68}; abstentions by case {'mixed': 46, 'feasible': 3, 'infeasible': 19}
- **H**: 960 structuring calls with a record; 0 model calls; changed/decoded slots 2006 ({'location_class': 623, 'artifact_class': 593, 'affordance_class': 533, 'parameter_name': 157, 'artifact_name': 54, 'affordance_name': 40, 'space_class': 2}); how {}; abstentions by case {}; mean tokens {'thinking_tokens': 358, 'answer_tokens': 187, 'free_answer_tokens': 183}; thinking cut and closed 0
- **H2**: 963 structuring calls with a record; 0 model calls; changed/decoded slots 2362 ({'location_class': 748, 'artifact_class': 658, 'affordance_class': 619, 'parameter_name': 176, 'artifact_name': 92, 'affordance_name': 69}); how {}; abstentions by case {'mixed': 50, 'feasible': 3, 'infeasible': 16}; mean tokens {'thinking_tokens': 370, 'answer_tokens': 204, 'free_answer_tokens': 184}; thinking cut and closed 4
- **G2M1**: 945 structuring calls with a record; 1187 model calls; changed/decoded slots 168 ({'parameter_name': 71, 'artifact_class': 65, 'affordance_name': 16, 'artifact_name': 10, 'affordance_class': 6}); how {'trie': 109, 'nearest': 4, 'abstain': 55}; abstentions by case {'mixed': 41, 'feasible': 2, 'infeasible': 12}
- **H2M1**: 973 structuring calls with a record; 0 model calls; changed/decoded slots 174 ({'parameter_name': 116, 'location_class': 16, 'artifact_class': 12, 'affordance_name': 12, 'artifact_name': 9, 'affordance_class': 9}); how {}; abstentions by case {'mixed': 48, 'infeasible': 11, 'feasible': 1}; mean tokens {'thinking_tokens': 898, 'answer_tokens': 170, 'free_answer_tokens': 175}; thinking cut and closed 12


#### Goals the structuring wrote (every goal of every structuring call)

| run | goals | no room | all 3 IRI slots null | no device class | no affordance class | `set` for a relative request |
|---|---|---|---|---|---|---|
| gpt5m | 989 | 8 | 2 | 130 | 192 | 308 |
| gpt5n | 987 | 33 | 24 | 59 | 106 | 349 |
| M1 | 910 | 6 | 2 | 60 | 105 | 311 |
| M2 | 913 | 12 | 8 | 67 | 103 | 314 |
| M4 | 982 | 36 | 19 | 24 | 46 | 462 |
| M3L | 981 | 41 | 20 | 26 | 46 | 443 |
| F | 959 | 32 | 20 | 35 | 73 | 434 |
| M3 | 983 | 29 | 14 | 22 | 32 | 471 |
| G | 955 | 30 | 11 | 15 | 28 | 456 |
| H | 949 | 243 | 133 | 154 | 177 | 413 |
| G2 | 951 | 33 | 14 | 88 | 92 | 458 |
| H2 | 967 | 101 | 76 | 135 | 145 | 418 |
| G2M1 | 954 | 15 | 9 | 130 | 175 | 330 |
| H2M1 | 982 | 18 | 18 | 116 | 118 | 336 |


Citire: H are 243 de scopuri fara camera si 133 cu toate cele 3 sloturi IRI `null` (din 949): `null` e o iesire ieftina pentru beam search. Regulile `no_null` ale H2 le reduc la **101 si 76 din 967** (10,4% / 7,9%) [masurat]:
mecanismul functioneaza *partial*. Pe cele 66 de episoade ale primei incercari reducerea fusese la 5,3% / 2,9%, deci pe tot setul gaura e mai mare decat parea. Ce a ramas nu mai e `null`: 100 de apeluri au
**raspunsul beam fara cheia camerei** (cheia lipseste cu totul), pentru cele 101 de scopuri fara camera (sectiunea 8.5). G2 are 33 / 14 si 88 de scopuri fara clasa dispozitivului (abstinenta: dispozitivul absent e raportat prin `unmatched_device`, nu scris).

---

## 7. Timp si cost

**Atentie la comparatii**: `workers` difera (M3 si G: 4; gpt-5: 4; H, G2, H2, G2-M1, H2-M1: 1; M1-M4, F: 1). Cu 4 workeri pe un GPU, timpul pe episod creste prin concurenta, desi timpul total scade.
Timpii pe episod nu se compara intre grupe de workeri; **numarul de apeluri si de tokeni** se compara. Durata reala a job-urilor de pe cluster (`sacct`, `Elapsed`):
M3 1:06:48 (4 workeri), G 1:56:15 (4), H 4:45:30 (1), G2 2:10:12 (1), **H2 5:03:22 (1, COMPLETED)**, prima incercare H2 4:33:08 (FAILED la 66 de episoade),
G2-M1 3:18:26 (1), H2-M1 4 joburi de 2:37:42 / 2:58:32 / 2:47:10 / 2:49:03 (1 worker fiecare; real ~8 h 23 min, doua in paralel). Rularile M1-M4, F, gpt-5 nu au job `sacct` (laptop / API).
Sinteza performanta x timp pentru toate rularile e in sectiunea 13; rularile locale partiale din 31 oct in sectiunea 14.

#### Time per episode (seconds, every record; includes failed calls)

| run | workers | records | mean | median | p95 | max | sum (h) | segmentation s | structuring s | planning s | execution s | other s | LLM calls | structuring s/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt5m | 4 | 400 | 16.3 | 14.2 | 31.1 | 66 | 1.81 | 5.5 | 6.7 | 0.3 | 0.1 | 2.2 | 4.2 | 5.8 |
| gpt5n | 4 | 400 | 16.8 | 15.2 | 29.0 | 45 | 1.87 | 4.9 | 7.9 | 0.7 | 0.1 | 1.8 | 4.4 | 7.0 |
| M1 | 1 | 400 | 61.6 | 41.5 | 192.0 | 229 | 6.85 | 11.6 | 44.5 | 0.3 | 0.1 | 3.6 | 4.2 | 47.5 |
| M2 | 1 | 400 | 76.3 | 61.3 | 200.9 | 267 | 8.48 | 10.2 | 40.5 | 0.5 | 0.1 | 23.6 | 4.2 | 44.4 |
| M4 | 1 | 400 | 38.6 | 27.2 | 93.9 | 271 | 4.29 | 12.9 | 20.9 | 1.7 | 0.0 | 1.6 | 3.5 | 21.6 |
| M3L | 1 | 400 | 35.3 | 27.0 | 82.6 | 195 | 3.92 | 12.5 | 20.8 | 0.4 | 0.0 | 0.2 | 3.5 | 21.6 |
| F | 1 | 400 | 49.7 | 39.3 | 112.0 | 263 | 5.52 | 17.6 | 28.6 | 1.3 | 0.0 | 0.1 | 3.5 | 17.5 |
| M3 | 4 | 400 | 33.8 | 25.1 | 79.5 | 263 | 3.76 | 10.9 | 19.3 | 0.8 | 0.0 | 0.1 | 3.6 | 10.9 |
| G | 4 | 400 | 61.4 | 44.0 | 177.2 | 277 | 6.82 | 14.7 | 37.8 | 4.3 | 0.3 | 1.1 | 18.5 | 3.4 |
| H | 1 | 400 | 41.5 | 28.2 | 122.9 | 322 | 4.61 | 4.3 | 34.7 | 0.8 | 0.0 | 0.2 | 4.7 | 22.3 |
| G2 | 1 | 400 | 18.5 | 10.7 | 42.7 | 200 | 2.06 | 4.3 | 11.5 | 0.9 | 0.1 | 0.2 | 14.5 | 1.3 |
| H2 | 1 | 400 | 44.0 | 29.4 | 130.4 | 330 | 4.89 | 4.3 | 37.2 | 0.8 | 0.1 | 0.2 | 4.8 | 24.0 |
| G2M1 | 1 | 400 | 28.8 | 13.9 | 183.9 | 252 | 3.20 | 4.3 | 22.3 | 0.5 | 0.0 | 0.2 | 4.9 | 9.6 |
| H2M1 | 1 | 400 | 99.3 | 58.5 | 309.0 | 977 | 11.04 | 7.8 | 89.6 | 0.2 | 0.0 | 0.3 | 4.2 | 57.7 |


#### Tokens per episode (mean; output includes reasoning tokens)

| run | total in | total out | structuring in | structuring out | structuring calls | failed LLM calls |
|---|---|---|---|---|---|---|
| gpt5m | 16712 | 1602 | 11746 | 1033 | 2.4 | 0 |
| gpt5n | 16972 | 2727 | 11917 | 1970 | 2.4 | 0 |
| M1 | 15714 | 2861 | 10796 | 2266 | 2.4 | 71 |
| M2 | 15656 | 5670 | 10781 | 2266 | 2.4 | 62 |
| M4 | 16113 | 2161 | 11744 | 1246 | 2.4 | 3 |
| M3L | 16105 | 1809 | 11731 | 1239 | 2.4 | 4 |
| F | 18578 | 1775 | 14223 | 1211 | 2.4 | 15 |
| M3 | 16184 | 1839 | 11809 | 1275 | 2.5 | 5 |
| G | 100505 | 2560 | 94090 | 1857 | 16.0 | 11 |
| H | 17560 | 2016 | 11598 | 1338 | 2.4 | 3 |
| G2 | 73023 | 2393 | 67093 | 1715 | 12.2 | 9 |
| H2 | 17802 | 2075 | 11598 | 1383 | 2.4 | 3 |
| G2M1 | 20903 | 3028 | 15883 | 2370 | 3.0 | 25 |
| H2M1 | 16790 | 4005 | 11785 | 2598 | 2.4 | 0 |


- **H2 vs H**: 44,0 s vs 41,5 s per episod (+6%), structurare 37,2 s vs 34,7 s, aceiasi tokeni (intrare 11.598 la ambele; iesire 1.383 vs 1.338). Mai mult timp la H2 vine din beam (6,1 s vs 5,2 s per apel) si din 4 gandiri taiate si inchise (`close_thinking`).
- **H2 vs G2**: 4,89 h vs 2,06 h (2,4x), dar G2 face 14,5 apeluri/episod si 73k tokeni de intrare, H2 4,8 si 17,8k. G2 e rapid pentru ca apelurile Ollama de decodare sunt scurte (1,3 s per apel), nu pentru ca face mai putin.
- **M1/M2** sunt cele mai lente (62 / 76 s per episod; structurarea Qwen3-8B gandeste ~2,3k tokeni); M2 are 23,6 s de "other" (raspunsul cu Qwen3-4B).
- **gpt-5-mini**: 16,3 s per episod, 16,7k tokeni de intrare, 1,6k de iesire (4 workeri).
- **Esecurile de apel**: M1 71 si M2 62 de apeluri esuate (timeout-uri Ollama local), M3 5, G 11, H 3, G2 9, **H2 3** (prima incercare: 13 in 66 de episoade). Harness-ul a logat 3 timeout-uri de segmentare (Ollama, nu serverul GENRE);
  le-am vazut in `.err`, dar nu le-am pus in corespondenta una la una cu cele 3 episoade scorate -1 (inregistrarile lor nu pastreaza eroarea).

### 7.1 Aceleasi 66 de episoade: inainte si dupa reparatia de viteza

#### Cost over the 66 episodes the first H2 attempt reached

| run | workers | mean wall s | structuring s | structuring s/call | failed calls | calls/episode |
|---|---|---|---|---|---|---|
| H | 1 | 60.7 | 54.3 | 27.0 | 0 | 6.1 |
| H2p | 1 | 243.8 | 237.6 | 120.9 | 13 | 6.0 |
| H2 | 1 | 61.4 | 54.8 | 27.6 | 0 | 6.3 |


### 7.2 Serverul GENRE (H si H2)

#### GENRE server logs (H, H2 first attempt, H2): requests, tokens, gaps, failures

| arm | requests answered | requests failed (client gave up, BrokenPipe) | prompt tok | thinking tok | free tok | answer tok | median gap s | p90 gap s |
|---|---|---|---|---|---|---|---|---|
| H | 960 | 0 | 4820 | 358 | 183 | 187 | 15 | 23 |
| H2p | 222 | 13 | 4879 | 353 | 181 | 195 | 68 | 91 |
| H2 | 963 | 0 | 4817 | 370 | 184 | 204 | 15 | 24 |
| H2M1 | 973 | 0 | 4845 | 898 | 175 | 170 | 28 | 62 |


#### GENRE masks and phases per structuring call (H, H2 first attempt, H2)

| arm | calls | json_steps | json_widened | filtered_steps | masked_steps | thinking s | free s | beam s (`genre`) | total s | beam median s | corr(json_widened, beam s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 960 | 653 | 9 | 0 | 108 | 5.7 | 3.0 | 5.2 | 14.0 | 4.7 | 0.02 |
| H2p | 209 | 672 | 244 | 286 | 127 | 6.2 | 3.3 | 59.9 | 69.4 | 57.7 | 0.49 |
| H2 | 963 | 758 | 17 | 287 | 125 | 6.0 | 3.1 | 6.1 | 15.1 | 4.8 | 0.48 |
| H2M1 | 973 | 572 | 5 | 271 | 119 | 23.5 | 4.6 | 8.4 | 36.5 | 7.0 | -0.02 |


Reparatia (`filter_ids`) a adus `json_widened` de la 244 la 17 per apel si beam-ul de la 59,9 s la 6,1 s. Mai ramane o corelatie de 0,48 intre `json_widened` si secundele de beam in interiorul lui H2 (la H, 0,02): tot mai exista pasi care largesc cautarea,
dar sunt putini (17 vs 9 la H) si nu mai conduc timpul.

---

## 8. H2: ce s-a intamplat

### 8.1 Rularea

Job **302815**, nod `ucsc-precis-h200-wn17`, 1 H200, `workers: 1`, pornit 2026-11-19 23:21, terminat 2026-11-20 04:21 (`sacct`: COMPLETED, 5:03:22, limita 23 h). Varianta `homebench.ecai.M3.pv-genre-v2.yaml`,
`GENRE_MODEL_DIR=~/hf_models/DeepSeek-R1-Distill-Qwen-7B`, 4 fascicule. 400 de inregistrari, 400 `.eval.json`, `stats/summary.md` scris de job; le-am copiat in `tests/homebench/results/harness/fep.genre-v2/` (cu log-urile in `_job/`)
si am rulat `error_report` local (41,3%, acelasi numar ca in tabelul de mai sus).

Despre commit: `run.json` si `summary.md` spun `22ad219c`, pentru ca repo-ul de pe fep nu a fost actualizat cu `git pull`. Cele doua fisiere schimbate de reparatie (`utils/genre_decoding.py`, `tests/simuhome/bench/genre_server.py`) au pe fep
acelasi continut ca in `c90490eb` (SHA-256 identic dupa normalizarea `\r\n`; verificat la scrierea acestui raport) [masurat]. Rularea e deci pe codul reparat, desi eticheta spune altceva.

Fara erori in rulare: `genre_server.log` fara `ERROR`/`Traceback`, 963 de cereri raspunse, 0 esuate; `.err` al harness-ului are doar erori obisnuite de executie (`HTTP 400` la actiuni gresite) si cele 3 timeout-uri de segmentare. Memorie: 45,1 GB cgroup la varf, 29,6 GB serverul, fara crestere (`mem.log`).

### 8.2 Prima incercare (job 300721) si cauza caderii ei

Prima incercare a lui H2 (job 300721, 17 nov) a ajuns FAILED (exit 120) dupa 4 h 33 min, cu 66 din 400 de episoade, 0 `.eval.json` si o inregistrare de 0 octeti. Nu era OOM (MaxRSS 55 GB), nici limita de timp, nici traceback in `.err`; ipoteza de atunci
(scrierea pe disc a esuat) **nu a fost verificata** si nu e confirmata acum. Nu stiu ce a cazut: rularea noua a mers cap-coada pe acelasi disc si acelasi nod-tip, ceea ce arata ca esecul nu era sistematic, dar nu il explica.
Rezultatele ei partiale sunt in `fep.genre-v2-partial/` (si pe fep in `...pv-genre-v2.partial-300721`); le folosesc doar pentru timpi (sectiunea 7.1).

### 8.3 De ce era lent si ce a reparat `filter_ids`

Pe prima incercare, beam search-ul sub masca dura 59,9 s per apel (H: 5,2 s) cu 244 de "largiri" ale cautarii per apel (`json_widened`, H: 9): filtrele `keys` / `no_null` respingeau aproape toti tokenii pe care modelul ii scoreaza sus, asa ca
procesorul de logits largea lista de candidati la tot vocabularul si evalua filtrul in Python pentru fiecare token. Commitul `c90490eb` ia tokenii permisi din trie (`HomeRules.filter_ids`) in loc sa parcurga vocabularul. Masurat dupa reparatie:
smoke de 7 episoade (job 302725): beam 9,4 s per apel, 77 s per episod; rularea completa: beam 6,1 s per apel, 44,0 s per episod, `json_widened` 17 [masurat]. Pe cele 66 de episoade ale primei incercari: 61,4 s per episod vs 243,8 s inainte.

### 8.4 Rezultatele lui H2

- **Fata de H** (acelasi mecanism, fara regulile v2): +9,3 puncte, 67 castigate / 30 pierdute (p < 0,001). Castig pe toate grupele: hb1 fezabile 64,9 -> 67,5%, hb1 infezabile 32,6 -> 59,3%, multi 20,6 -> 24,5%, mixed 4,2 -> 11,6%.
  Cele patru reguli sunt introduse impreuna, deci nu stiu cat din castig vine de la fiecare; `infra_llm_error` 16 -> 0 e compatibil cu `close_thinking`, dar nu e izolat de un brat doar cu el [ipoteza].
- **Fata de G2** (aceeasi idee de abstinenta, alt mecanism): 41,3% vs 43,8%, nu se distinge (p = 0,32). 46 de episoade trec doar la H2, 57 doar la G2 (anexa A). Pe tipuri (coloana "case" din anexa): G2 trece si H2 pica la 30 fezabile / 16 infezabile / 11 mixed; H2 trece si G2 pica la 24 fezabile / 16 infezabile / 6 mixed.
  Pe infezabile sunt deci la egalitate (16 / 16; H2 are mai putin `infeas_executed_other`, 6 vs 21, dar mai mult `infeas_substituted`, 21 vs 17); G2 castiga pe fezabile si mixed (cheile omise, sectiunea 8.5).
- **Fata de gpt-5-mini**: 11 castigate / 179 pierdute (p < 0,001). H2 e la jumatate din rata unui model de frontiera mic si mult sub M1 (72,7%).
- **Costul**: H2 foloseste cu 4x mai putini tokeni de intrare decat G2 (17,8k vs 73k per episod) si e de 2,4x mai lent.

### 8.5 Gaura ramasa: cheile omise

Un scop dintr-un raspuns GENRE are trei sloturi IRI: `location_class`, `artifact_class`, `affordance_class`. Regulile v2 spun: `no_null` = pe un slot de camera / dispozitiv / actiune nu se poate scrie `null` cat timp casa are valori;
`keys` = o cheie JSON ramane prefixul unei chei scrise de raspunsul liber (`utils/genre_decoding.py`, docstring). Nicio regula nu spune ca o cheie *trebuie* scrisa, iar `no_null` se aplica numai dupa ce cheia exista.
Beam search-ul, care alege secventa intreaga cu scor maxim, a gasit iesirea urmatoare: sa nu scrie cheia deloc.

#### GENRE answers that omit a room / device / action key

| arm | structuring calls | calls missing a key | missing room | missing device class | missing action class | key was in the free answer (any) | episodes affected | affected: pass | other episodes: pass |
|---|---|---|---|---|---|---|---|---|---|
| H | 960 | 64 (6.7%) | 57 | 46 | 53 | 58 | 60 | 3/60 = 5.0% | 124/337 = 36.8% |
| H2 | 963 | 110 (11.4%) | 100 | 60 | 77 | 105 | 90 | 11/90 = 12.2% | 153/307 = 49.8% |
| H2M1 | 973 | 65 (6.7%) | 18 | 61 | 62 | 65 | 63 | 44/63 = 69.8% | 253/337 = 75.1% |


Exemple (episoade, raspunsul liber vs beam) [masurat]: `hb1_feasible_seed_3969` (cerere: "Set the temperature of the air conditioner in the study room to 23 degrees"): raspunsul liber are `StudyRoom`, `AirConditioner`, `SetAbsoluteLevelCommand`;
raspunsul beam nu are niciuna din cele trei chei, deci scopul iese cu toate trei `null`. `hb1_feasible_seed_4908`: raspunsul beam are clasa dispozitivului si a actiunii, dar nu `location_class`.

Citire: [masurat] 100 din 110 de apeluri cu cheie lipsa pierd camera; in 105 din 110 cel putin una din cheile lipsa era in raspunsul liber, deci modelul stia valoarea. Episoadele atinse (90, 23% din cele scorate) trec 12,2%, celelalte 49,8%;
dintre cele 22 de esecuri `seg_part_lost` ale lui H2, 20 sunt in episoade atinse. Fenomenul exista si la H (60 de episoade, 5,0% trec), deci nu l-a creat v2, dar l-a **crescut** de la 6,7% la 11,4% din apeluri:
inchiderea iesirii `null` a mutat o parte din beam-uri pe iesirea "cheie lipsa". [ipoteza ca mecanism; cifrele sunt masurate]

Efectul posibil: daca cele 90 de episoade ar trece la rata celorlalte (49,8%), H2 ar avea ~198 / 397 = ~50% [calculat; estimare-margine, pentru ca episoadele atinse sunt mai grele: 44 din 90 sunt mixed]. Nu am rulat nicio reparatie.

Reparatie propusa (nu facuta): in `HomeRules`, o cheie IRI a schemei care lipseste din raspunsul beam sa nu poata fi inchisa (nu se permite `}` inainte ca cele trei chei sa fi aparut cand raspunsul liber le-a avut),
sau completarea cheilor lipsa din raspunsul liber prin acelasi `abstention.overlay`. Costa o rulare noua de ~5 h pe fep ("H3"); cere decizia ta.

### 8.6 Ce nu stiu

- Daca diferenta G2 / H2 de 2,5 puncte (sau cea pe multi / mixed) e reala: un singur sir de rulare per brat, p = 0,32.
- Cat din castigul H -> H2 vine din fiecare regula (nu exista brate de ablatie).
- Daca H2 fara cheile omise ar depasi G2 (estimarea de 50% e o margine superioara).
- Cauza caderii primei incercari (sectiunea 8.2).

---

## 9. Ce explica diferentele dintre experimente

1. **De la M3 la M1 (+50 puncte): modelul de structurare.** Acelasi prompt, aceleasi episoade; DS-R1-Distill-Qwen-7B scrie clasele fara prefix (403 de parti de notatie) si valori in campul vecin; Qwen3-8B face ~30. 207 episoade trec doar cu Qwen3-8B.
2. **De la M3 la G (+12 puncte): notatia reparata prin trie, platita cu infezabile.** Partile fezabile 12% -> 55%; infezabile 96% -> 57% pentru ca trie-ul inlocuieste un dispozitiv absent cu cel mai apropiat din camera (`infeas_substituted` 33). Cost: 16 apeluri si 6x tokeni.
3. **De la G la H (-2, nesemnificativ): alt backend, o singura trecere, dar `null`.** Cele mai bune fezabile (64,9% hb1), cele mai slabe infezabile (32,6%); beam search alege `null` (ieftin) si chei cu cratima; 62 de parti `seg_part_lost`, 70 `struct_room_missing`, 16 erori de infrastructura.
   H nu e comparabil curat cu M3 (alt backend/ponderi/template); un control `decoding: free` pe HF nu exista.
4. **De la G la G2 (+9,5 puncte, p < 0,001): evidenta si abstinenta.** `struct_wrong_affordance` 64 -> 12, `infeas_substituted` 33 -> 17, 1.515 apeluri de model mai putine in total (6.386 -> 4.871). Efecte secundare: `infeas_executed_other` 10 -> 21 si `struct_false_clarify_device` 2 -> 14
   (regula evidentei schimba camera corecta; abstinenta are fals pozitive: aspiratorul). Cele doua schimbari sunt amestecate; nu exista un brat doar cu una.
5. **De la H la H2 (+9,3 puncte, p < 0,001): `no_null` + `keys` + abstinenta + inchiderea gandirii.** `struct_room_missing` 70 -> 29, `seg_part_lost` 62 -> 30, `plan_unrouted` 41 -> 11, `infra_llm_error` 16 -> 0, hb1 infezabile 32,6 -> 59,3%. Ramane: cheile omise (11,4% din apeluri) si valorile gresite (108 parti).
6. **F (JSON) e un pas inapoi (-3,4 puncte, p = 0,013).** Repara prefixul (234 -> 43) dar modelul copiaza valori din campul vecin: `struct_wrong_room` 82 -> 239, `struct_invalid_device_name` 38 -> 124. Formatul contextului nu e parghia.
7. **Planificatorul nu conteaza** (M3 vs M4, M1 vs M2): planificarea explicita e SPARQL deterministic pe numele si clasele structurii.
8. **G2 vs H2: doua mecanisme, acelasi rezultat.** Prin Ollama cu ~12 apeluri (G2) sau in-process cu un apel si beam search (H2), 41-44% pe ECAI-400, cu modelul de structurare ramas DS-R1-7B. Tavanul vine de la model si de la `struct_wrong_value`, nu de la mecanismul de constrangere.
9. **G2-M1 / H2-M1 vs M1: cu Qwen3-8B decodarea nu se vede.** 76,3% / 74,2% fata de 72,7% (p = 0,12 / 0,65), in banda de zgomot a lui M1 vs M2 (32 / 27). Decodarea elimina notatia (31 -> 0 de parti) si ridica partile fezabile (84,1% -> 90,6% / 88,7%), dar a intervenit de ~12x mai rar decat la DeepSeek (168-174 de sloturi fata de 1932-2362), iar infezabilele nu se misca (79,9% -> 79,4% / 81,7%). Controlul nu e curat (laptop vs fep), sectiunea 12.5.

---

## 10. Limite si confuzii (ce nu se poate concluziona)

- **O singura rulare per brat**, `temperature: 0`; zgomotul intre rulari "identice" e ~4% din episoade (M3L -> M3). Diferentele sub ~+/- 8 episoade din 400 sunt zgomot.
- **Backend-uri diferite**: M1-M4, F, G, G2 ruleaza prin Ollama (ponderi cuantizate, `ds-r1-7b-16k`), H si H2 prin Hugging Face (bf16). H si H2 se compara intre ele, cu M3 doar cu rezerva.
- **Workeri si noduri diferite**: M3 si G pe 4 workeri, restul pe 1; M3 pe nodul wn15, G/H/G2 pe wn12, H2 pe wn17; M1-M4, F pe masina locala (Ollama pe localhost). Timpii nu sunt comparabili intre aceste grupe.
- **Comituri diferite** (coloana `commit`): G si H au rulat codul `bf3be605`, G2 `22ad219c`, H2 `c90490eb` (desi `run.json` spune `22ad219c`, sectiunea 8.1). G -> G2 si H -> H2 amesteca cod nou cu zgomotul unei a doua rulari.
- **Patru reguli introduse deodata la H2**: nu exista ablatie; nu se poate atribui castigul H -> H2 unei reguli anume.
- **Scorul principal are o podea** (sectiunea 1) si **etichete suspecte** (657 de randuri `unexist_attribute`; 147 din 309 cereri fezabile/mixed cer o schimbare "cu N procente", unde gold-ul aduna puncte): plafonul oricarui model e sub 100%.
- **Corelatia cheie omisa / esec** (12% vs 50%) e masurata, dar episoadele atinse sunt mai multe-parti (mai grele); nu e o masura a castigului reparatiei.
- **G2-M1 / H2-M1 vs M1**: controlul M1 a rulat pe laptop (Ollama local, commit `68e51755`, 31 oct, 15 episoade de infrastructura, 71 de apeluri cazute prin timeout); bratele noi pe fep (H200), cod `0de6ea3f` / `e15d5ca1`, alt nod (G2-M1 pe wn17; H2-M1: s1 pe wn17, s2-s4 pe wn12).
  H2-M1 foloseste Hugging Face bf16 la structurare, controlul Ollama cuantizat. Perechile M1 -> brat sunt deci "alta masina + alt backend + decodare" amestecate, iar zgomotul masurat intre rulari apropiate (M1 vs M2, 32 / 27) e de acelasi ordin cu efectul.
  Seturile de episoade scorate difera (M1 scoate 15, G2-M1 8, H2-M1 0); perechile folosesc doar episoadele scorate in ambele rulari (378 si 385 din 400).
- **H2-M1 e unit din patru joburi** (100 de episoade fiecare, `ATTEMPT=s1..s4`); `run.json`-ul unit e scris de mine din cele patru (episoadele concatenate, cheia `merged_from`), iar `stats/` e regenerat pe folderul unit, nu luat de la joburi.
- **Verificari pe care nu le-am facut**: nu am rulat niciun model, nu am profilat serverul GENRE, nu am rulat suita de teste (nu am schimbat cod de produs; am modificat un script de analiza), nu am gasit cauza caderii primei incercari.

---

## 11. Cum reproduci

```bash
# aduce rezultatele H2 de pe fep (doar citire); stats/ e director, cere -r
scp -r "fep:experiments/semantic/harness/1/homebench.ecai.M3.pv-genre-v2/." tests/homebench/results/harness/fep.genre-v2/
scp fep:experiments/semantic/harness/jobs/302815/genre_server.log fep:experiments/semantic/harness/jobs/302815/mem.log \
    fep:llm-agents-for-ami/ami_harness_302815.out fep:llm-agents-for-ami/ami_harness_302815.err tests/homebench/results/harness/fep.genre-v2/_job/

# raportul de erori al unei rulari (scrie in <run>/stats/)
PYTHONUTF8=1 venv/Scripts/python.exe -m tests.homebench.error_report tests/homebench/results/harness/fep.genre-v2

# G2-M1 (un job) si H2-M1 (patru shard-uri, unite local); doar citire
scp -r "fep:experiments/semantic/harness/1/homebench.ecai.M1.pv-trie-v2/." tests/homebench/results/harness/fep.trie-v2-m1/
for i in 1 2 3 4; do scp -r "fep:experiments/semantic/harness/s$i/homebench.ecai.M1.pv-genre-v2/." tests/homebench/results/harness/fep.genre-v2-m1/_shards/s$i/; done
cp -n tests/homebench/results/harness/fep.genre-v2-m1/_shards/s*/hb*.json tests/homebench/results/harness/fep.genre-v2-m1/   # 400 inregistrari + 400 .eval.json
#   run.json unit: episoadele celor 4 run.json concatenate + cheia merged_from (sectiunea 12.1)
python -m tests.simuhome.bench.stats --records tests/homebench/results/harness/fep.genre-v2-m1
python -m tests.homebench.error_report tests/homebench/results/harness/fep.genre-v2-m1   # si fep.trie-v2-m1

# raportul de erori al tuturor rularilor complete (regenereaza <run>/stats/; fara LLM)
for r in homebench.ecai.gpt-5-mini homebench.ecai.gpt-5-nano homebench.ecai.M1 homebench.ecai.M2 homebench.ecai.M3 homebench.ecai.M4          homebench.ecai.M3.pv-json-v2 fep400.M3 fep.trie-v1 fep.genre-v1 fep.trie-v2 fep.genre-v2 fep.trie-v2-m1 fep.genre-v2-m1; do
  PYTHONUTF8=1 venv/Scripts/python.exe -m tests.homebench.error_report tests/homebench/results/harness/$r
done

# toate tabelele acestui raport (din fisierele existente, fara LLM)
PYTHONUTF8=1 venv/Scripts/python.exe -I work/reports/experiments_summary.py > tables.md
# timpul rularilor locale (sectiunea 14)
PYTHONUTF8=1 venv/Scripts/python.exe -I work/reports/timing_analysis.py
```

Ce am facut in aceasta sesiune: am verificat pe fep (`sacct`, `ls`, doar citire) ca jobul 302815 s-a terminat, am copiat rezultatele si log-urile local, am rulat `error_report` pe H2, am actualizat
`work/reports/experiments_summary.py` (H2 complet, perechi, tabelul cheilor omise, dezacordurile G2 / H2) si am regenerat acest raport. Pe fep: doar citire (`sacct`, `ls`, `scp`, comparatie SHA-256 a doua fisiere); nu am lansat si nu am anulat niciun job.

Decizii care sunt ale tale (nu le-am luat): un brat H3 care interzice cheile omise (~5 h de GPU, sectiunea 8.5), un brat doar cu abstinenta sau doar cu evidenta, un control `decoding: free` pe HF, regula `set -> modify` pentru cererile relative,
o a doua rulare a lui G2 si H2 pentru a vedea zgomotul (singura cale de a ordona G2 si H2).

---

## 12. G2-M1 si H2-M1: decodarea constransa cu Qwen3-8B la structurare

Intrebarea: decodarea care a ridicat M3 (DeepSeek-R1-7B) de la 22,6% la 41-44% ridica si un model care deja scrie notatia corecta? Controlul este M1 (structurare Qwen3-8B, text liber, 72,7%).
Tabelele de mai sus (sectiunile 2, 3, 3.1, 5, 6, 7, 8.5) contin acum si liniile `G2M1` / `H2M1`; aici e ce spun ele.

### 12.1 Rularile

- **G2-M1**: job **303338**, nodul `ucsc-precis-h200-wn17`, 1 H200, 1 worker, 2026-11-20 16:26 - 19:44, **3 h 18 min 26 s**, `COMPLETED`, exit 0:0 [masurat, `sacct`]. Config `homebench.ecai.M1.pv-trie-v2.yaml`, commit `0de6ea3f`.
  Structurarea e facuta de `qwen3-8b-16k` pe Ollama; cererile au limita de 180 s ca la control. 400 de inregistrari si 400 `.eval.json`.
- **H2-M1**: patru joburi **303360-303363** (`ATTEMPT=s1..s4`, cate 100 de episoade, impartite dupa numarul de comenzi), toate `COMPLETED`, exit 0:0: 2:37:42, 2:58:32, 2:47:10, 2:49:03 [masurat, `sacct`]; 2026-11-20 16:58 - 2026-11-21 01:21.
  s1 pe `wn17`, s2-s4 pe `wn12`. Commit `e15d5ca1`, `GENRE_MAX_TOKENS=10240`, `BENCH_REQUEST_TIMEOUT=1500`, ponderile Qwen3-8B descarcate pe `/tmp`-ul nodului (`GENRE_HF_REPO`). QOS-ul permite doar doua joburi simultane, deci au rulat doua cate doua.
- **Unirea shard-urilor** (facuta in aceasta sesiune, local): cele 4 foldere `s1..s4` (cate 200 de fisiere: 100 de inregistrari + 100 `.eval.json`) au fost copiate in `tests/homebench/results/harness/fep.genre-v2-m1/`
  (400 + 400, fara duplicate; originalele raman in `_shards/`, log-urile joburilor in `_job/<job>/`). `run.json` unit = episoadele celor patru, `merged_from` cu jobul si ora de start a fiecarui shard.
  Apoi `tests.simuhome.bench.stats` si `tests.homebench.error_report` pe folderul unit (fara apeluri LLM): 297 trecute din 400, acelasi numar ca la concatenarea directa a celor patru `episodes.csv`. Pe fep, folderele `s1..s4` nu au fost atinse.
- G2-M1 pe fep: `homebench.ecai.M1.pv-trie-v2` -> local `fep.trie-v2-m1/` (cu `_job/`: `mem.log`, config, `.out` / `.err`). Log-urile `ollama.log` (16 MB) nu au fost copiate.

### 12.2 Rezultate

| run | scorate | trecute | Wilson 95% | hb1 fezabile | hb1 infezabile | multi fezabile | mixed | parti fezabile | parti infezabile |
|---|---|---|---|---|---|---|---|---|---|
| M1 (control, laptop) | 385 | 280 = 72,7% | 68-77 | 89,9% | 85,2% | 66,7% | 48,4% | 84,1% | 79,9% |
| G2-M1 | 392 | 299 = 76,3% | 72-80 | 91,9% | 84,3% | 76,7% | 50,5% | 90,6% | 79,4% |
| H2-M1 | 400 | 297 = 74,2% | 70-78 | 90,4% | 87,2% | 71,2% | 46,9% | 88,7% | 81,7% |
| gpt-5-mini (referinta) | 400 | 334 = 83,5% | 80-87 | 99,1% | 83,7% | 87,5% | 60,4% | 96,0% | 79,8% |

Perechi (acelasi episod, infra scoase; tabelul complet in 3.1) [calculat]:

| pereche | n | castigate | pierdute | p |
|---|---|---|---|---|
| M1 -> G2-M1 | 378 | 42 | 28 | 0,120 |
| M1 -> H2-M1 | 385 | 41 | 36 | 0,649 |
| G2-M1 -> H2-M1 | 392 | 27 | 36 | 0,314 |
| gpt-5-mini -> G2-M1 | 392 | 14 | 43 | < 0,001 |
| gpt-5-mini -> H2-M1 | 400 | 20 | 57 | < 0,001 |
| G2 -> G2-M1 (alt model de structurare) | 387 | 141 | 12 | < 0,001 |
| H2 -> H2-M1 (alt model de structurare) | 397 | 147 | 14 | < 0,001 |

Concluzie: nicio diferenta intre M1, G2-M1 si H2-M1 nu depaseste zgomotul; ambele brate raman sub gpt-5-mini (p < 0,001). Cea mai mare diferenta, G2-M1 vs M1 (+3,6 puncte, 42 / 28), are p = 0,12 si ar fi putut iesi din oricare doua rulari ale lui M1
(M1 vs M2: 32 / 27). Cele doua mecanisme nu se disting nici intre ele (p = 0,31), ca la DeepSeek (G2 vs H2: p = 0,32).

### 12.3 Ce a schimbat decodarea la Qwen3 si ce nu [masurat, tabelele din sectiunile 5 si 6]

- **Notatia**: 31 de parti picate la M1 din familia "notatie", 29 din ele `struct_parameter_name`; 0 la G2-M1 si 0 la H2-M1. La DeepSeek aceeasi familie avea 403 de parti. Asta e singura eroare pe care decodarea a eliminat-o, si la Qwen3 era mica.
- **Partile fezabile**: 84,1% (M1) -> 90,6% (G2-M1) / 88,7% (H2-M1). Castigul vine din hb1 fezabile (89,9% -> 91,9% / 90,4%) si multi fezabile (66,7% -> 76,7% / 71,2%); mixed nu se misca (48,4% -> 50,5% / 46,9%).
- **Partile infezabile**: 79,9% -> 79,4% / 81,7%, fara schimbare. Familia "parte infezabila gestionata gresit" are 37 de parti la M1, 60 la G2-M1 si 61 la H2-M1: `infeas_substituted` 21 -> 27 / 24, `struct_false_clarify_attribute` 10 -> 15 / 29,
  `struct_false_clarify_device` 6 -> 14 / 8. Aceste numere se compara cu grija: M1 are 15 episoade scoase ca infrastructura (962 de parti in episoade scorate, fata de 979 si 991), deci seturile nu sunt identice.
  Abstinenta a actionat (G2-M1: 55 de abstineri, 41 in mixed, 12 in infezabile, 2 in fezabile; H2-M1: 48 / 11 / 1; in episoadele fezabile, unde o abstinere e sigur gresita, 2 si 1), dar substitutiile ramase arata ca nu a acoperit toate cazurile.
- **Episoade pierdute pe stadii** (primul stadiu care a pierdut): structurare 54 (M1) / 34 (G2-M1) / 55 (H2-M1); fezabilitate 23 / 25 / 22; executie 9 / 11 / 10; planificare 1 / 1 / 9 (H2-M1: `plan_unrouted` 7, `plan_missing_entry` 2); segmentare 3 / 3 / 3.
- **`struct_false_clarify_attribute` la H2-M1 (29 de parti, 28 de episoade, fata de 10 la M1)**: am verificat un episod, `hb1_feasible_seed_3241` ("Increase the speed of the fan ... by 1 level"): M1, G2-M1 si H2-M1 scriu **acelasi** scop
  (`affordance_name: speed`, `parameter_name: NA`) si decodarea nu intervine, deci mesajul "nu are setarea ..." vine din raspunsul modelului, nu din decodare. Cauza celor 28 de episoade nu e stabilita;
  suprapunerea cu M1 e mica (6 episoade comune), cu G2-M1 de 11. [ipoteza: alta cale numerica la bf16 pe Hugging Face fata de Ollama cuantizat; nemasurata]
- **Cererile relative**: Qwen3 scrie `goal_effect: modify` la 35 din 38 de cereri relative hb1 fezabile la M1 (33 trec), 37 / 40 la G2-M1 (34 trec), 40 / 43 la H2-M1 (37 trec). Problema care costa bratele cu DeepSeek (`set` pentru o cerere relativa: 0-1 din 20-24 trec) nu exista la Qwen3.
- **Cat a intervenit decodarea**: G2-M1: 945 de apeluri de structurare cu inregistrare, 1187 de apeluri la model (G2: 4871), 168 de sloturi schimbate (`parameter_name` 71, `artifact_class` 65, `affordance_name` 16), `trie` 109, `nearest` 4, `abstain` 55.
  H2-M1: 973 de apeluri, 174 de sloturi (`parameter_name` 116, `location_class` 16, `artifact_class` 12). Cu DeepSeek: 1932 (G2) si 2362 (H2).
- **Cheile omise la GENRE** (8.5): 65 din 973 de apeluri (6,7%, ca la H; 11,4% la H2), dintre care 18 fara camera. Cele 63 de episoade atinse trec **44 / 63 = 69,8%**, celelalte 253 / 337 = 75,1%. La H2 cu DeepSeek: 12,2% vs 49,8%.
  La Qwen3 gaura exista (in toate cele 65 de cazuri cheia lipsa fusese scrisa de raspunsul liber) dar nu pare sa coste episoade [masurat; diferenta de 5 puncte, n = 63, nu am testat-o].

### 12.4 Timp si cost [masurat]

| brat | s / episod (mediu) | median | p95 | max | suma (h) | structurare s / apel | apeluri de structurare / episod | apeluri LLM esuate |
|---|---|---|---|---|---|---|---|---|
| M1 (laptop) | 61,6 | 41,5 | 192 | 229 | 6,85 | 47,5 | 2,4 | 71 |
| G2-M1 (fep) | 28,8 | 13,9 | 184 | 252 | 3,20 | 9,6 | 3,0 | 25 |
| H2-M1 (fep, HF) | 99,3 | 58,5 | 309 | 977 | 11,04 | 57,7 | 2,4 | 0 |
| H2 (DeepSeek, fep) | 44,0 | 29,4 | 130 | 330 | 4,89 | 24,0 | 2,4 | 3 |

Timpii nu se compara intre masini (M1 pe laptop). Pe aceeasi masina: H2-M1 e de 3,4x mai lent decat G2-M1 pe episod. Serverul GENRE al lui H2-M1: 973 de cereri raspunse, 0 esuate, mediana intre cereri 28 s (p90 62 s);
pe apel: gandire 23,5 s + raspuns liber 4,6 s + beam 8,4 s = 36,5 s, cu gandire de 898 de tokeni in medie; gandirea a fost taiata si inchisa (`close_thinking`) in 12 apeluri.
Cele 13 apeluri de smoke (jobs 303339 / 303357) dadusera 1,9k tokeni in medie si ~94 s per apel, de unde estimarea de ~25 h pentru un singur job: masurat pe cele 400 de episoade, suma timpilor e de 11,0 h, sub limita de 23 h.
Impartirea in patru joburi a fost deci mai prudenta decat era nevoie (un singur job ar fi incaput, [calculat]); a scurtat timpul real la ~8 h 23 min (doua joburi in paralel).

Infrastructura: G2-M1 are 8 episoade de infrastructura (13 parti `infra_llm_timeout`, cererea de 180 s ca la M1, a carui infrastructura a fost 37 de parti) iar H2-M1 are 0 (`BENCH_REQUEST_TIMEOUT=1500`). Episoadele de infrastructura sunt scoase din rate si din perechi,
deci H2-M1 e scorat pe 400 de episoade, G2-M1 pe 392 si M1 pe 385: **seturile difera** (episoadele care dau timeout sunt cele cu gandire lunga, probabil mai grele; sensul si marimea efectului nu au fost masurate).

### 12.5 Ce nu se poate concluziona

- **Efectul decodarii pe Qwen3** nu e masurat curat: controlul M1 a rulat pe laptop, pe alt cod (`68e51755`) si cu Ollama local; G2-M1 / H2-M1 pe fep. Intre M3 local si M3 fep, acelasi cod si model, 8 / 8 episoade isi schimbau verdictul; intre M1 si M2, 32 / 27.
  Un +3,6 puncte (G2-M1) sau +1,5 (H2-M1) nu iese din aceasta banda.
- **H2-M1 foloseste alt backend la structurare** decat controlul (Hugging Face bf16 fata de Ollama cuantizat), asa cum H2 fata de M3; nu exista un control `decoding: free` pe Hugging Face.
- **G2-M1 vs H2-M1** (p = 0,31): nu se poate spune care mecanism e mai bun la Qwen3; H2-M1 costa de 3,4x mai mult timp.
- Cauza celor 28 de episoade `struct_false_clarify_attribute` la H2-M1 si cea a substitutiilor ramase la infezabile nu sunt stabilite (12.3).

### 12.6 Ce ar urma (decizii ale tale, nimic din asta nu e rulat)

1. **Control curat**: M1 pe fep, acelasi cod si aceeasi infrastructura ca G2-M1 (`decoding: free`, Ollama), 400 de episoade; dupa G2-M1, cam 3-4 h [estimare, nemasurata]. Singura cale de a spune daca +3,6 puncte e real.
2. **Control `decoding: free` pe Hugging Face** pentru H2-M1 (cum e M3 pe HF pentru H2), ca sa izolezi efectul backend-ului; ~10 h dupa H2-M1 [estimare].
3. Daca tinta e un sistem care se descurca pe infezabile cu Qwen3, partea rezolvabila nu e notatia: `infeas_substituted` (24-27 de parti) si `struct_false_clarify_*` (32 de parti la G2-M1, 37 la H2-M1) raman, indiferent de decodare.

---

## 13. Performanta si timp: toate rularile complete intr-un loc

Aceasta sectiune pune scorul si timpul alaturi, pentru cele 14 rulari complete. Tabelul e generat de `experiments_summary.py` (functia `efficiency_table`), din aceleasi inregistrari ca sectiunile 3 si 7.
"Suma timpilor pe episod" nu e durata reala cand `workers` > 1 (M3, G, gpt-5: 4 workeri); pentru rularile de pe cluster, coloana "cluster job hours" e suma `Elapsed` din `sacct`
(la H2-M1: cele patru joburi adunate, nu cele ~8 h 23 min reale, pentru ca doua au rulat in paralel).

#### Performance and time of every full run (400 episodes)

| run | scored | passed | pass % | sum of episode times (h) | s / episode | hours per passed episode | passed per hour | cluster job hours | passed per job hour | structuring calls / episode | tokens in / episode | tokens out / episode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt5m | 400 | 334 | 83.5% | 1.81 | 16.3 | 0.005 | 184.8 | - | - | 2.4 | 16712 | 1602 |
| gpt5n | 400 | 295 | 73.8% | 1.87 | 16.8 | 0.006 | 158.0 | - | - | 2.4 | 16972 | 2727 |
| M1 | 385 | 280 | 72.7% | 6.85 | 61.6 | 0.024 | 40.9 | - | - | 2.4 | 15714 | 2861 |
| M2 | 388 | 291 | 75.0% | 8.48 | 76.3 | 0.029 | 34.3 | - | - | 2.4 | 15656 | 5670 |
| M4 | 399 | 90 | 22.6% | 4.29 | 38.6 | 0.048 | 21.0 | - | - | 2.4 | 16113 | 2161 |
| M3L | 399 | 91 | 22.8% | 3.92 | 35.3 | 0.043 | 23.2 | - | - | 2.4 | 16105 | 1809 |
| F | 396 | 76 | 19.2% | 5.52 | 49.7 | 0.073 | 13.8 | - | - | 2.4 | 18578 | 1775 |
| M3 | 399 | 90 | 22.6% | 3.76 | 33.8 | 0.042 | 23.9 | 1.11 | 80.8 | 2.5 | 16184 | 1839 |
| G | 394 | 135 | 34.3% | 6.82 | 61.4 | 0.050 | 19.8 | 1.94 | 69.7 | 16.0 | 100505 | 2560 |
| H | 397 | 127 | 32.0% | 4.61 | 41.5 | 0.036 | 27.5 | 4.76 | 26.7 | 2.4 | 17560 | 2016 |
| G2 | 393 | 172 | 43.8% | 2.06 | 18.5 | 0.012 | 83.5 | 2.17 | 79.3 | 12.2 | 73023 | 2393 |
| H2 | 397 | 164 | 41.3% | 4.89 | 44.0 | 0.030 | 33.5 | 5.06 | 32.4 | 2.4 | 17802 | 2075 |
| G2M1 | 392 | 299 | 76.3% | 3.20 | 28.8 | 0.011 | 93.5 | 3.31 | 90.4 | 3.0 | 20903 | 3028 |
| H2M1 | 400 | 297 | 74.2% | 11.04 | 99.3 | 0.037 | 26.9 | 11.21 | 26.5 | 2.4 | 16790 | 4005 |

Cum se citeste [calculat din tabel]:

- **Trei grupe de hardware, necomparabile la timp**: laptop (M1-M4, F, Ollama pe RTX 4070 SUPER, 1 worker), cluster (M3, G, H, G2, H2, G2-M1, H2-M1, H200, 1-4 workeri) si API (gpt-5-mini / nano, 4 workeri). In interiorul unei grupe, "passed per hour" si "hours per passed episode" se pot compara.
- **Pe cluster, cele mai multe episoade trecute pe ora de job**: G2-M1 90,4, M3 80,8, G2 79,3, G 69,7; apoi H2 32,4, H 26,7, H2-M1 26,5. Doua familii: decodarea prin Ollama cu apeluri scurte (G, G2, G2-M1) si GENRE in-process cu gandire lunga (H, H2, H2-M1).
- **Cu aceeasi decodare, scorul urca la fel de mult cu Qwen3-8B ca fara ea, dar timpul nu**: G2-M1 (76,3%) face 28,8 s per episod, H2-M1 (74,2%) face 99,3 s, adica de 3,4x mai mult pentru o diferenta de 2,1 puncte care nu e semnificativa (p = 0,31, sectiunea 12.2).
  Pe DeepSeek aceeasi pereche: G2 18,5 s vs H2 44,0 s (2,4x), 43,8% vs 41,3% (p = 0,32).
- **Ce plateste decodarea fata de M3**: G2 adauga 21,2 puncte (22,6% -> 43,8%) cu ~2x timpul de job (1,11 h -> 2,17 h; M3 a rulat pe 4 workeri, G2 pe 1) si 4,5x tokeni de intrare; H2 adauga 18,7 puncte cu ~4,5x timpul de job (5,06 h) si aceiasi tokeni de intrare (17,8k).
  Pe Qwen3-8B (M1 -> G2-M1): +3,6 puncte (in zgomot) si 3,0 apeluri de structurare per episod fata de 2,4, deci decodarea costa putin acolo unde modelul scrie deja notatia corecta, dar nici nu aduce mult.
- **Tokeni**: G are 100,5k de tokeni de intrare per episod (16 apeluri de structurare), G2 73,0k (12,2 apeluri), G2-M1 20,9k (3,0); restul bratelor 15,7-18,6k (2,4 apeluri). Tokenii de iesire cresc la Qwen3-8B (2,9k la M1, 4,0k la H2-M1: gandirea) fata de DeepSeek (1,8-2,6k).
- **F (JSON)** e cel mai slab pe ambele axe: 19,2% si 13,8 episoade trecute pe ora, sub M3 (22,6% si 23,9).
- **gpt-5-mini** e cel mai bun pe ambele (83,5%, 16,3 s per episod pe 4 workeri), dar nu intr-o comparatie de hardware: ruleaza pe API.

Ce nu spune tabelul: costul in bani al rularilor gpt-5 (nu e masurat in acest raport) si timpul de pe cluster al rularilor laptop (nu au rulat acolo).

---

## 14. Rularile locale (laptop): ce se stie despre timp si ce s-a schimbat fata de analiza din 31 octombrie

`work/reports/analiza-timp-rulari-locale.md` (scrisa pentru Alex Sorici pe 31 oct) analiza rulari **partiale**: M1 cu 138 de episoade (00:48-08:18, segmentare DS-R1-1.5B), M3 cu 21, `qwen2.5-coder-7b` cu 50, `qwen3-8b` cu 20.
Folderul `homebench.ecai.M1` contine acum rularea completa pornita la 2026-10-31 13:10 (segmentare `qwen3-8b-16k`, 400 de episoade), iar `homebench.ecai.M3` pe cea de la 1 nov (400 de episoade); [inferenta din `run.json` si din orele din documentul vechi: incercarile partiale au fost inlocuite in aceleasi foldere].
De aceea **cifrele documentului vechi pentru M1 si M3 nu se mai regenereaza**: `timing_analysis.py` pe folderele de acum da alte numere. Documentul vechi ramane valabil ca snapshot al incercarilor partiale (M1 / M3 de atunci) si, nemodificat, pentru `qwen2.5-coder-7b` si `qwen3-8b` (folderele lor nu s-au schimbat: 50, respectiv 20 de episoade).
Rezultatul generat acum (`timing_analysis.py`, fara LLM, pe folderele curente) [masurat]:

#### Per run

| run | episodes | total h | mean s | median s | p90 s | max s | simulator hangs | hang h |
|---|---|---|---|---|---|---|---|---|
| M1 | 400 | 6.85 | 62 | 41 | 162 | 229 | 0 | 0.00 |
| M3 | 400 | 3.92 | 35 | 27 | 64 | 195 | 0 | 0.00 |
| coder-7b | 50 | 0.33 | 24 | 16 | 64 | 75 | 0 | 0.00 |
| qwen3-8b | 20 | 0.32 | 57 | 56 | 123 | 158 | 0 | 0.00 |

#### M1: where its wall time went (all finished episodes)

| part | hours | share |
|---|---|---|
| structuring, other episodes | 3.64 | 53.2% |
| structuring, episodes with a timed-out call | 1.31 | 19.1% |
| segmentation | 1.29 | 18.9% |
| planning, profiling, guarding, response, other, execution (normal) | 0.44 | 6.5% |
| unaccounted (harness, discovery, between stages) | 0.16 | 2.3% |
| simulator hang (execution stage of the hung episodes) | 0.00 | 0.0% |
| total | 6.85 | 100% |

Without the 0 hung episodes: 400 episodes, mean 62 s, median 41 s, 6.85 h; at that mean 400 episodes = 6.8 h.

#### Seconds per episode by stage (all finished episodes)

| run | segmentation | structuring | planning | profiling | guarding | response | execution | other | wall |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 11.6 | 44.5 | 0.3 | 0.0 | 0.0 | 0.0 | 0.1 | 3.6 | 61.6 |
| M3 | 12.5 | 20.8 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 35.3 |
| coder-7b | 6.2 | 14.8 | 0.4 | 0.0 | 0.0 | 0.0 | 0.1 | 1.2 | 24.0 |
| qwen3-8b | 8.7 | 40.7 | 2.1 | 0.0 | 0.0 | 0.0 | 0.1 | 3.7 | 56.9 |

#### Cost of one structuring call when the episode has a single call (no queue), by model

| run | model | calls | mean tok in | mean tok out | median s | mean s | p90 s |
|---|---|---|---|---|---|---|---|
| M1 | qwen3-8b-16k | 192 | 4938 | 1092 | 13.4 | 16.7 | 31.6 |
| M3 | ds-r1-7b-16k | 202 | 4969 | 527 | 10.0 | 10.3 | 12.0 |
| coder-7b | qwen2.5-coder-7b-16k | 20 | 6198 | 198 | 4.3 | 4.3 | 4.8 |
| qwen3-8b | qwen3-8b-16k | 10 | 4426 | 1390 | 11.1 | 20.3 | 55.4 |

#### Why the structuring prompt differs: the segmenter's qualifiers decide the route

| run | intents | simple route | dependency: specificity only, no kind | dependency: qualifiers empty | dependency: dependency label | dependency: maintenance | dependency: other |
|---|---|---|---|---|---|---|---|
| M1 | 977 | 969 (99%) | 0 (0%) | 0 (0%) | 8 (1%) | 0 (0%) | 0 (0%) |
| M3 | 976 | 967 (99%) | 0 (0%) | 0 (0%) | 9 (1%) | 0 (0%) | 0 (0%) |
| coder-7b | 159 | 48 (30%) | 0 (0%) | 0 (0%) | 111 (70%) | 0 (0%) | 0 (0%) |
| qwen3-8b | 49 | 49 (100%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |

Structuring calls by route (call and intent matched by order and text; seconds are for calls that did not time out):

| run | route | calls | timed out | rate | mean s (ok) | mean tok in (ok) | mean tok out (ok) |
|---|---|---|---|---|---|---|---|
| M1 | dependency | 8 | 2 | 25% | 52.6 | 9137 | 1326 |
| M1 | simple | 969 | 68 | 7% | 37.1 | 4732 | 997 |
| M3 | dependency | 9 | 1 | 11% | 35.1 | 8777 | 808 |
| M3 | simple | 967 | 2 | 0% | 21.0 | 4790 | 507 |
| coder-7b | dependency | 111 | 1 | 1% | 16.5 | 9122 | 201 |
| coder-7b | simple | 48 | 0 | 0% | 12.2 | 4584 | 189 |
| qwen3-8b | simple | 49 | 0 | 0% | 36.8 | 4525 | 1107 |

Diferente fata de documentul din 31 oct [masurat, comparatie intre cele doua):

| afirmatie in analiza din 31 oct (M1 partial, 138 de episoade) | rularea completa M1 (400 de episoade, folderul de acum) |
|---|---|
| 60% din timp = doua episoade blocate in `execution` (hang simulator, 4,5 h din 7,45 h) | 0 hang-uri; `execution` 0,1 s per episod; 6,85 h in total |
| 49 din 270 de apeluri de structurare taiate la 180 s, toate pe ruta "dependency" (segmentator DS-R1-1.5B fara eticheta de tip) | 70 din 977 de apeluri taiate (7,2%): 68 pe ruta "simple", 2 pe "dependency"; 99% din intenturi merg pe "simple" (segmentator Qwen3-8B) |
| structurarea e 89% din timp | structurarea e 72% (4,95 h din 6,85 h), segmentarea 18,9% (1,29 h) |
| schimbarea modelelor in VRAM 1,6% | nemasurabil acum: logul Ollama din acea zi nu mai e pe disc (scriptul raporteaza 0 incarcari) |

Concluzii care raman [masurat]: structurarea domina timpul la modelele cu gandire; un apel de structurare izolat (fara coada) costa 16,7 s la Qwen3-8B (1092 de tokeni de iesire), 10,3 s la DS-R1-7B (527), 4,3 s la Qwen2.5-Coder-7B (198);
cu `OLLAMA_NUM_PARALLEL=1` apelurile unui episod se pun la coada (suma secundelor pe apel e de ~2,6x durata structurarii la M1). Concluzia ruta gresita => timeout nu mai explica M1-ul complet: cu segmentator Qwen3-8B rutarea e corecta, dar timeout-urile (7,2%) raman,
deci cauza lor in rularea completa nu e stabilita [ipoteza: generari lungi ale lui Qwen3-8B, ca in documentul vechi (mediana 8827 de tokeni la apelurile taiate), nemasurata pe rularea completa]. Recomandarea 4 din documentul vechi (timeout mai mare) exista acum ca `BENCH_REQUEST_TIMEOUT` (commit `5f6e8425`): H2-M1 a rulat cu 1500 s si 0 episoade de infrastructura, G2-M1 a ramas la 180 s si are 8 (sectiunea 12.4). Recomandarea 3 (`max_tokens` la structurare) nu e aplicata la bratele Ollama; la GENRE exista plafonul `GENRE_MAX_TOKENS` (10240 la H2-M1).

---

## Anexa A. Episoadele pe care G2 si H2 difera

Perechea cea mai buna a fiecarui mecanism. Coloanele M3 / G / H / G2 / H2 dau verdictul si primul cod de eroare.

#### Episodes G2 passes and H2 fails (57)

| episode | case | parts | M3 | G | H | G2 | H2 |
|---|---|---|---|---|---|---|---|
| hb1_feasible_seed_1106_fixed | feasible | 1 | fail struct_wrong_room | fail struct_wrong_value | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb1_feasible_seed_1749 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb1_feasible_seed_2269 | feasible | 1 | fail struct_invalid_device_name | pass | fail struct_room_missing | pass | fail exec_wrong_value |
| hb1_feasible_seed_2328 | feasible | 1 | fail struct_class_prefix | pass | fail struct_room_missing | pass | fail struct_wrong_affordance |
| hb1_feasible_seed_2815 | feasible | 1 | fail struct_class_prefix | pass | fail struct_room_missing | pass | fail struct_wrong_value |
| hb1_feasible_seed_2843 | feasible | 1 | fail struct_slot_confusion | pass | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb1_feasible_seed_3550 | feasible | 1 | fail struct_invalid_device_name | fail struct_wrong_value | fail struct_room_missing | pass | fail struct_false_clarify_device |
| hb1_feasible_seed_3575 | feasible | 1 | fail struct_invalid_device_name | fail struct_wrong_affordance | pass | pass | fail struct_wrong_affordance |
| hb1_feasible_seed_3969 | feasible | 1 | fail struct_class_prefix | pass | pass | pass | fail struct_room_missing |
| hb1_feasible_seed_4480 | feasible | 1 | fail struct_slot_confusion | fail struct_wrong_affordance | pass | pass | fail struct_wrong_device_class |
| hb1_feasible_seed_4966 | feasible | 1 | pass | pass | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_5104 | feasible | 1 | fail struct_class_prefix | pass | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb1_feasible_seed_6127 | feasible | 1 | fail struct_invalid_device_name | fail exec_wrong_value | fail struct_room_missing | pass | fail struct_wrong_value |
| hb2_feasible_seed_106 | feasible | 2 | fail struct_class_prefix | fail struct_wrong_value | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb2_feasible_seed_111 | feasible | 2 | fail struct_class_prefix | fail exec_wrong_value | fail struct_room_missing | pass | fail struct_wrong_value |
| hb2_feasible_seed_122_fixed | feasible | 2 | fail struct_class_prefix | pass | fail struct_wrong_affordance | pass | fail struct_wrong_affordance |
| hb2_feasible_seed_124 | feasible | 2 | fail struct_class_prefix | fail struct_wrong_affordance | pass | pass | fail struct_wrong_value |
| hb2_feasible_seed_28 | feasible | 2 | fail struct_class_prefix | pass | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb2_feasible_seed_29 | feasible | 2 | fail struct_room_missing | pass | fail exec_wrong_value | pass | fail plan_false_impossible |
| hb2_feasible_seed_32 | feasible | 2 | fail struct_wrong_room | fail struct_wrong_room | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb2_feasible_seed_41 | feasible | 2 | fail struct_class_prefix | fail struct_wrong_value | fail struct_wrong_affordance | pass | fail struct_wrong_value |
| hb2_feasible_seed_43 | feasible | 2 | fail struct_class_prefix | pass | pass | pass | fail plan_false_impossible |
| hb2_feasible_seed_7 | feasible | 2 | fail struct_class_prefix | pass | pass | pass | fail struct_wrong_value |
| hb2_feasible_seed_70 | feasible | 2 | fail struct_class_prefix | fail struct_wrong_affordance | fail struct_wrong_affordance | pass | fail struct_wrong_affordance |
| hb2_feasible_seed_73 | feasible | 2 | fail struct_class_prefix | fail struct_wrong_value | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb3_feasible_seed_10 | feasible | 3 | fail struct_class_prefix | pass | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb3_feasible_seed_41 | feasible | 3 | fail struct_class_prefix | pass | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb3_feasible_seed_46 | feasible | 3 | fail struct_class_prefix | fail struct_value_missing | fail struct_room_missing | pass | fail struct_room_missing |
| hb3_feasible_seed_8 | feasible | 3 | fail struct_class_prefix | fail plan_explicit_failed | fail struct_value_missing | pass | fail plan_explicit_failed |
| hb4_feasible_seed_25 | feasible | 4 | fail struct_slot_confusion | fail struct_wrong_value | pass | pass | fail exec_wrong_value |
| hb1_infeasible_seed_1644 | infeasible | 1 | pass | pass | fail infra_llm_error | pass | fail seg_part_lost |
| hb1_infeasible_seed_2545 | infeasible | 1 | pass | pass | fail seg_part_lost | pass | fail seg_part_lost |
| hb1_infeasible_seed_2700 | infeasible | 1 | pass | fail plan_explicit_failed | pass | pass | fail plan_explicit_failed |
| hb1_infeasible_seed_2721 | infeasible | 1 | pass | pass | fail seg_part_lost | pass | fail infeas_accepted_on_target |
| hb1_infeasible_seed_3053 | infeasible | 1 | pass | pass | pass | pass | fail infeas_substituted |
| hb1_infeasible_seed_3582 | infeasible | 1 | pass | pass | fail seg_part_lost | pass | fail infeas_substituted |
| hb1_infeasible_seed_3619 | infeasible | 1 | pass | pass | fail plan_unrouted | pass | fail seg_part_lost |
| hb1_infeasible_seed_364 | infeasible | 1 | pass | fail infeas_executed_other | fail seg_part_lost | pass | fail seg_part_lost |
| hb1_infeasible_seed_38 | infeasible | 1 | pass | infra infra_llm_timeout | pass | pass | fail infeas_accepted_on_target |
| hb1_infeasible_seed_4080 | infeasible | 1 | fail plan_unrouted | fail plan_unrouted | fail plan_unrouted | pass | fail infeas_substituted |
| hb1_infeasible_seed_4644 | infeasible | 1 | pass | fail infeas_substituted | fail infeas_substituted | pass | fail seg_part_lost |
| hb1_infeasible_seed_4940 | infeasible | 1 | pass | fail seg_part_lost | fail plan_unrouted | pass | fail seg_part_lost |
| hb1_infeasible_seed_5093 | infeasible | 1 | pass | fail infeas_executed_other | pass | pass | fail infeas_substituted |
| hb1_infeasible_seed_5550 | infeasible | 1 | pass | pass | fail infeas_executed_other | pass | fail seg_part_lost |
| hb1_infeasible_seed_5721 | infeasible | 1 | pass | pass | fail seg_part_lost | pass | fail seg_part_lost |
| hb1_infeasible_seed_6312 | infeasible | 1 | pass | fail infeas_substituted | fail plan_unrouted | pass | fail infeas_substituted |
| hb2_mixed_seed_104 | mixed | 2 | fail struct_class_prefix | pass | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb2_mixed_seed_136 | mixed | 2 | fail struct_wrong_value | pass | fail infeas_executed_other | pass | fail struct_wrong_value |
| hb2_mixed_seed_138 | mixed | 2 | pass | pass | fail plan_unrouted | pass | fail infeas_accepted_on_target |
| hb2_mixed_seed_194 | mixed | 2 | fail struct_class_prefix | fail seg_part_lost | fail seg_part_lost | pass | fail seg_part_lost |
| hb2_mixed_seed_34 | mixed | 2 | fail struct_slot_confusion | pass | fail plan_unrouted | pass | fail exec_wrong_value |
| hb2_mixed_seed_35 | mixed | 2 | fail struct_class_prefix | fail struct_wrong_value | fail seg_part_lost | pass | fail plan_unrouted |
| hb2_mixed_seed_36 | mixed | 2 | fail struct_wrong_value | fail struct_wrong_value | fail seg_part_lost | pass | fail exec_wrong_value |
| hb3_mixed_seed_109 | mixed | 3 | fail struct_invalid_device_name | fail infra_llm_timeout | fail infra_llm_error | pass | fail seg_part_lost |
| hb3_mixed_seed_124 | mixed | 3 | fail struct_class_prefix | fail seg_part_lost | fail plan_false_impossible | pass | fail plan_false_impossible |
| hb4_mixed_seed_80 | mixed | 4 | fail struct_class_prefix | fail struct_wrong_value | fail exec_wrong_value | pass | fail struct_room_missing |
| hb5_mixed_seed_450 | mixed | 5 | fail struct_slot_confusion | fail struct_wrong_affordance | fail exec_wrong_value | pass | fail struct_false_clarify_device |


#### Episodes H2 passes and G2 fails (46)

| episode | case | parts | M3 | G | H | G2 | H2 |
|---|---|---|---|---|---|---|---|
| hb1_feasible_seed_1094 | feasible | 1 | fail struct_class_prefix | fail struct_wrong_device_class | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb1_feasible_seed_2088 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb1_feasible_seed_2113 | feasible | 1 | fail struct_placeholder_echo | fail plan_false_impossible | pass | fail plan_false_impossible | pass |
| hb1_feasible_seed_3264 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | fail exec_wrong_value | fail struct_wrong_value | pass |
| hb1_feasible_seed_3337 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_3460 | feasible | 1 | fail struct_slot_confusion | fail struct_wrong_value | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_3844 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | fail struct_room_missing | fail struct_wrong_value | pass |
| hb1_feasible_seed_3915 | feasible | 1 | fail struct_room_missing | fail exec_wrong_value | fail exec_wrong_value | fail struct_room_missing | pass |
| hb1_feasible_seed_395 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_affordance | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_4151 | feasible | 1 | fail struct_class_prefix | fail plan_false_impossible | pass | fail plan_false_impossible | pass |
| hb1_feasible_seed_4383 | feasible | 1 | fail struct_slot_confusion | fail exec_wrong_value | pass | fail exec_wrong_value | pass |
| hb1_feasible_seed_460 | feasible | 1 | fail struct_slot_confusion | pass | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_4806 | feasible | 1 | fail struct_class_prefix | fail exec_wrong_value | fail exec_wrong_value | fail exec_wrong_value | pass |
| hb1_feasible_seed_6134 | feasible | 1 | fail struct_class_prefix | fail exec_wrong_value | fail infra_llm_error | fail exec_wrong_value | pass |
| hb1_feasible_seed_789 | feasible | 1 | fail struct_class_prefix | pass | pass | fail exec_wrong_value | pass |
| hb2_feasible_seed_101 | feasible | 2 | fail struct_class_prefix | fail struct_wrong_value | fail struct_room_missing | fail struct_wrong_value | pass |
| hb2_feasible_seed_114 | feasible | 2 | fail struct_wrong_room | fail struct_wrong_device_class | fail struct_wrong_affordance | fail struct_wrong_device_class | pass |
| hb2_feasible_seed_117 | feasible | 2 | fail struct_slot_confusion | pass | fail struct_wrong_affordance | fail struct_wrong_value | pass |
| hb2_feasible_seed_118 | feasible | 2 | fail struct_value_missing | pass | pass | fail exec_wrong_value | pass |
| hb2_feasible_seed_119 | feasible | 2 | fail struct_class_prefix | fail plan_false_impossible | pass | fail plan_false_impossible | pass |
| hb2_feasible_seed_14 | feasible | 2 | fail struct_slot_confusion | fail struct_wrong_value | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb2_feasible_seed_20 | feasible | 2 | fail struct_class_prefix | pass | pass | fail struct_value_missing | pass |
| hb2_feasible_seed_58_fixed | feasible | 2 | fail struct_slot_confusion | fail struct_wrong_affordance | pass | fail struct_wrong_value | pass |
| hb3_feasible_seed_38 | feasible | 3 | fail struct_slot_confusion | fail struct_wrong_value | fail struct_wrong_value | fail plan_false_impossible | pass |
| hb1_infeasible_seed_1858 | infeasible | 1 | pass | fail plan_explicit_failed | pass | fail infeas_accepted_on_target | pass |
| hb1_infeasible_seed_2595 | infeasible | 1 | pass | pass | fail plan_explicit_failed | fail infeas_executed_other | pass |
| hb1_infeasible_seed_3613 | infeasible | 1 | pass | pass | fail infeas_accepted_on_target | fail seg_part_lost | pass |
| hb1_infeasible_seed_3640 | infeasible | 1 | pass | fail infeas_accepted_on_target | fail plan_unrouted | fail plan_explicit_failed | pass |
| hb1_infeasible_seed_3740 | infeasible | 1 | pass | pass | fail infra_llm_error | fail infeas_executed_other | pass |
| hb1_infeasible_seed_4138 | infeasible | 1 | pass | pass | fail infra_llm_error | fail infeas_executed_other | pass |
| hb1_infeasible_seed_4532 | infeasible | 1 | fail seg_part_lost | fail plan_unrouted | pass | fail infeas_substituted | pass |
| hb1_infeasible_seed_4786 | infeasible | 1 | fail plan_unrouted | pass | pass | fail struct_json_invalid | pass |
| hb1_infeasible_seed_5287 | infeasible | 1 | fail plan_unrouted | fail plan_unrouted | pass | fail plan_unrouted | pass |
| hb1_infeasible_seed_5673 | infeasible | 1 | pass | fail infeas_executed_other | pass | fail infeas_executed_other | pass |
| hb1_infeasible_seed_6158 | infeasible | 1 | pass | pass | fail infeas_executed_other | fail infeas_executed_other | pass |
| hb1_infeasible_seed_6238 | infeasible | 1 | pass | fail infeas_executed_other | fail seg_part_lost | fail infeas_executed_other | pass |
| hb1_infeasible_seed_6482 | infeasible | 1 | pass | fail plan_explicit_failed | pass | fail plan_explicit_failed | pass |
| hb1_infeasible_seed_6657 | infeasible | 1 | pass | fail infeas_substituted | pass | fail infeas_substituted | pass |
| hb1_infeasible_seed_831 | infeasible | 1 | pass | fail infeas_substituted | fail plan_unrouted | fail infeas_substituted | pass |
| hb1_infeasible_seed_862 | infeasible | 1 | pass | pass | pass | fail plan_explicit_failed | pass |
| hb2_mixed_seed_137 | mixed | 2 | fail struct_class_prefix | pass | pass | fail infeas_executed_other | pass |
| hb2_mixed_seed_215 | mixed | 2 | fail struct_invalid_device_name | fail exec_wrong_value | fail struct_room_missing | fail struct_wrong_value | pass |
| hb2_mixed_seed_84 | mixed | 2 | fail struct_class_prefix | fail plan_unrouted | fail plan_unrouted | fail infeas_substituted | pass |
| hb3_mixed_seed_159 | mixed | 3 | fail struct_slot_confusion | fail struct_wrong_value | fail infeas_executed_other | fail struct_wrong_value | pass |
| hb3_mixed_seed_289 | mixed | 3 | fail struct_class_prefix | fail struct_wrong_affordance | fail seg_part_lost | fail plan_explicit_failed | pass |
| hb3_mixed_seed_6 | mixed | 3 | fail struct_class_prefix | fail struct_wrong_value | fail struct_wrong_value | fail struct_value_missing | pass |


## Anexa B. Rularile vechi nescorate de acest raport

M5, `qwen3-8b` si `qwen2.5-coder-7b` (partiale, fara `.eval.json`; timpii lor sunt in sectiunea 14) si `fep400.pv-trie` (copie identica a `fep.trie-v1`, acelasi job 299621) nu apar in tabele.
`homebench.ecai.M3.trie-set20` si `pv-trie` / `pv-trie.v1` / `pv-repair` sunt rularile de 20 de episoade din sectiunea 4.3.

---

## Anexa C. Episoadele pe care M1, G2-M1 si H2-M1 difera

Perechile din sectiunea 12.2 (coloanele M1 / G2M1 / H2M1 dau verdictul si primul cod de eroare; generate de `experiments_summary.py`).

#### Episodes M1 passes and G2M1 fails (28)

| episode | case | parts | M1 | G2M1 | H2M1 |
|---|---|---|---|---|---|
| hb1_feasible_seed_2853 | feasible | 1 | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_3241 | feasible | 1 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb1_feasible_seed_4578 | feasible | 1 | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_5104 | feasible | 1 | pass | fail exec_wrong_value | pass |
| hb2_feasible_seed_95 | feasible | 2 | pass | fail exec_wrong_value | pass |
| hb3_feasible_seed_2_fixed | feasible | 3 | pass | fail seg_part_lost | fail struct_wrong_value |
| hb3_feasible_seed_31 | feasible | 3 | pass | fail struct_false_clarify_attribute | pass |
| hb3_feasible_seed_36 | feasible | 3 | pass | fail infra_llm_timeout | pass |
| hb4_feasible_seed_15 | feasible | 4 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb4_feasible_seed_21 | feasible | 4 | pass | fail infra_llm_timeout | fail struct_false_clarify_attribute |
| hb5_feasible_seed_10 | feasible | 5 | pass | fail infra_llm_timeout | fail struct_wrong_room |
| hb1_infeasible_seed_1701 | infeasible | 1 | pass | fail infeas_substituted | pass |
| hb1_infeasible_seed_2456 | infeasible | 1 | pass | fail infeas_substituted | fail infeas_substituted |
| hb1_infeasible_seed_2686 | infeasible | 1 | pass | fail infeas_substituted | fail infeas_substituted |
| hb1_infeasible_seed_3034 | infeasible | 1 | pass | fail infeas_substituted | pass |
| hb1_infeasible_seed_5751 | infeasible | 1 | pass | fail infeas_substituted | pass |
| hb2_mixed_seed_135 | mixed | 2 | pass | fail exec_wrong_value | fail exec_wrong_value |
| hb2_mixed_seed_35 | mixed | 2 | pass | fail infeas_substituted | fail struct_false_clarify_attribute |
| hb3_mixed_seed_342 | mixed | 3 | pass | fail infra_llm_timeout | pass |
| hb3_mixed_seed_370 | mixed | 3 | pass | fail struct_wrong_value | fail infeas_substituted |
| hb4_mixed_seed_153 | mixed | 4 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb4_mixed_seed_283 | mixed | 4 | pass | fail struct_false_clarify_device | fail seg_part_lost |
| hb4_mixed_seed_299 | mixed | 4 | pass | fail struct_false_clarify_device | fail struct_false_clarify_device |
| hb5_mixed_seed_36 | mixed | 5 | pass | fail struct_false_clarify_device | pass |
| hb5_mixed_seed_4 | mixed | 5 | pass | fail struct_false_clarify_device | pass |
| hb6_mixed_seed_86 | mixed | 6 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb8_mixed_seed_407 | mixed | 8 | pass | fail infra_llm_timeout | pass |
| hb9_mixed_seed_314_fixed | mixed | 9 | pass | fail struct_wrong_value | fail struct_false_clarify_attribute |

#### Episodes G2M1 passes and M1 fails (42)

| episode | case | parts | M1 | G2M1 | H2M1 |
|---|---|---|---|---|---|
| hb1_feasible_seed_2269 | feasible | 1 | fail struct_parameter_name | pass | pass |
| hb1_feasible_seed_3264 | feasible | 1 | fail exec_wrong_value | pass | fail struct_false_clarify_attribute |
| hb1_feasible_seed_3297 | feasible | 1 | fail struct_wrong_value | pass | pass |
| hb1_feasible_seed_4383 | feasible | 1 | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb1_feasible_seed_5201 | feasible | 1 | fail struct_parameter_name | pass | pass |
| hb1_feasible_seed_5307 | feasible | 1 | fail struct_wrong_value | pass | pass |
| hb2_feasible_seed_101 | feasible | 2 | fail struct_wrong_value | pass | pass |
| hb2_feasible_seed_129 | feasible | 2 | fail struct_parameter_name | pass | pass |
| hb2_feasible_seed_41 | feasible | 2 | fail struct_parameter_name | pass | pass |
| hb2_feasible_seed_42 | feasible | 2 | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb2_feasible_seed_45 | feasible | 2 | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb2_feasible_seed_88 | feasible | 2 | fail struct_parameter_name | pass | pass |
| hb3_feasible_seed_18 | feasible | 3 | fail exec_wrong_value | pass | pass |
| hb3_feasible_seed_28 | feasible | 3 | fail exec_wrong_value | pass | fail struct_wrong_value |
| hb3_feasible_seed_39 | feasible | 3 | fail seg_part_lost | pass | fail plan_missing_entry |
| hb3_feasible_seed_6 | feasible | 3 | fail struct_parameter_name | pass | pass |
| hb3_feasible_seed_8 | feasible | 3 | fail struct_parameter_name | pass | pass |
| hb4_feasible_seed_10 | feasible | 4 | fail struct_wrong_value | pass | pass |
| hb4_feasible_seed_11 | feasible | 4 | fail struct_parameter_name | pass | fail struct_wrong_value |
| hb4_feasible_seed_14 | feasible | 4 | fail struct_false_clarify_attribute | pass | fail struct_false_clarify_attribute |
| hb4_feasible_seed_25 | feasible | 4 | fail struct_wrong_value | pass | pass |
| hb7_feasible_seed_2 | feasible | 7 | fail infra_llm_timeout | pass | pass |
| hb7_feasible_seed_6 | feasible | 7 | fail struct_parameter_name | pass | fail struct_wrong_value |
| hb8_feasible_seed_1 | feasible | 8 | fail infra_llm_timeout | pass | fail struct_wrong_value |
| hb1_infeasible_seed_240 | infeasible | 1 | fail infeas_substituted | pass | fail infeas_substituted |
| hb1_infeasible_seed_4134 | infeasible | 1 | fail infeas_substituted | pass | pass |
| hb1_infeasible_seed_6312 | infeasible | 1 | fail infeas_substituted | pass | pass |
| hb2_mixed_seed_175 | mixed | 2 | fail infeas_accepted_on_target | pass | fail infeas_accepted_on_target |
| hb2_mixed_seed_194 | mixed | 2 | fail struct_parameter_name | pass | pass |
| hb2_mixed_seed_236 | mixed | 2 | fail struct_parameter_name | pass | pass |
| hb2_mixed_seed_34 | mixed | 2 | fail struct_parameter_name | pass | pass |
| hb3_mixed_seed_147 | mixed | 3 | fail struct_json_invalid | pass | pass |
| hb3_mixed_seed_289 | mixed | 3 | fail struct_parameter_name | pass | pass |
| hb3_mixed_seed_325 | mixed | 3 | fail struct_parameter_name | pass | pass |
| hb3_mixed_seed_42 | mixed | 3 | fail struct_parameter_name | pass | pass |
| hb4_mixed_seed_191_fixed | mixed | 4 | fail infra_llm_timeout | pass | pass |
| hb4_mixed_seed_414 | mixed | 4 | fail infra_llm_timeout | pass | fail struct_wrong_value |
| hb4_mixed_seed_80 | mixed | 4 | fail infra_llm_timeout | pass | pass |
| hb5_mixed_seed_450 | mixed | 5 | fail struct_parameter_name | pass | fail plan_unrouted |
| hb7_mixed_seed_251 | mixed | 7 | fail struct_parameter_name | pass | fail struct_false_clarify_attribute |
| hb7_mixed_seed_4 | mixed | 7 | fail struct_parameter_name | pass | pass |
| hb9_mixed_seed_236 | mixed | 9 | fail infra_llm_timeout | pass | pass |

#### Episodes G2M1 passes and H2M1 fails (36)

| episode | case | parts | M1 | G2M1 | H2M1 |
|---|---|---|---|---|---|
| hb1_feasible_seed_3264 | feasible | 1 | fail exec_wrong_value | pass | fail struct_false_clarify_attribute |
| hb1_feasible_seed_3324 | feasible | 1 | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_3460 | feasible | 1 | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_3844 | feasible | 1 | pass | pass | fail struct_false_clarify_attribute |
| hb1_feasible_seed_4057 | feasible | 1 | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_4383 | feasible | 1 | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb1_feasible_seed_4509_fixed | feasible | 1 | pass | pass | fail plan_unrouted |
| hb1_feasible_seed_5478 | feasible | 1 | pass | pass | fail struct_wrong_device_class |
| hb2_feasible_seed_42 | feasible | 2 | fail exec_wrong_value | pass | fail exec_wrong_value |
| hb2_feasible_seed_45 | feasible | 2 | fail struct_wrong_value | pass | fail struct_wrong_value |
| hb2_feasible_seed_48 | feasible | 2 | pass | pass | fail struct_false_clarify_attribute |
| hb3_feasible_seed_28 | feasible | 3 | fail exec_wrong_value | pass | fail struct_wrong_value |
| hb3_feasible_seed_32 | feasible | 3 | pass | pass | fail exec_wrong_value |
| hb3_feasible_seed_38 | feasible | 3 | pass | pass | fail struct_false_clarify_attribute |
| hb3_feasible_seed_39 | feasible | 3 | fail seg_part_lost | pass | fail plan_missing_entry |
| hb3_feasible_seed_9 | feasible | 3 | pass | pass | fail struct_false_clarify_attribute |
| hb4_feasible_seed_11 | feasible | 4 | fail struct_parameter_name | pass | fail struct_wrong_value |
| hb4_feasible_seed_14 | feasible | 4 | fail struct_false_clarify_attribute | pass | fail struct_false_clarify_attribute |
| hb5_feasible_seed_7 | feasible | 5 | pass | pass | fail struct_false_clarify_attribute |
| hb6_feasible_seed_2 | feasible | 6 | pass | pass | fail plan_unrouted |
| hb7_feasible_seed_6 | feasible | 7 | fail struct_parameter_name | pass | fail struct_wrong_value |
| hb8_feasible_seed_1 | feasible | 8 | fail infra_llm_timeout | pass | fail struct_wrong_value |
| hb1_infeasible_seed_240 | infeasible | 1 | fail infeas_substituted | pass | fail infeas_substituted |
| hb1_infeasible_seed_5913 | infeasible | 1 | infra infra_llm_timeout | pass | fail infeas_substituted |
| hb1_infeasible_seed_6238 | infeasible | 1 | pass | pass | fail infeas_substituted |
| hb2_mixed_seed_175 | mixed | 2 | fail infeas_accepted_on_target | pass | fail infeas_accepted_on_target |
| hb2_mixed_seed_178 | mixed | 2 | pass | pass | fail plan_explicit_failed |
| hb2_mixed_seed_36 | mixed | 2 | pass | pass | fail struct_false_clarify_attribute |
| hb3_mixed_seed_253 | mixed | 3 | pass | pass | fail plan_unrouted |
| hb3_mixed_seed_52_fixed | mixed | 3 | pass | pass | fail infeas_accepted_on_target |
| hb4_mixed_seed_414 | mixed | 4 | fail infra_llm_timeout | pass | fail struct_wrong_value |
| hb4_mixed_seed_79 | mixed | 4 | pass | pass | fail plan_unrouted |
| hb5_mixed_seed_239 | mixed | 5 | pass | pass | fail struct_false_clarify_attribute |
| hb5_mixed_seed_450 | mixed | 5 | fail struct_parameter_name | pass | fail plan_unrouted |
| hb7_mixed_seed_251 | mixed | 7 | fail struct_parameter_name | pass | fail struct_false_clarify_attribute |
| hb8_mixed_seed_245 | mixed | 8 | pass | pass | fail struct_wrong_value |

#### Episodes H2M1 passes and G2M1 fails (27)

| episode | case | parts | M1 | G2M1 | H2M1 |
|---|---|---|---|---|---|
| hb1_feasible_seed_2853 | feasible | 1 | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_3550 | feasible | 1 | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute | pass |
| hb1_feasible_seed_4578 | feasible | 1 | pass | fail struct_wrong_value | pass |
| hb1_feasible_seed_4611 | feasible | 1 | fail exec_wrong_value | fail exec_wrong_value | pass |
| hb1_feasible_seed_5104 | feasible | 1 | pass | fail exec_wrong_value | pass |
| hb1_feasible_seed_5715 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb2_feasible_seed_31 | feasible | 2 | fail struct_false_clarify_device | fail struct_false_clarify_other | pass |
| hb2_feasible_seed_32 | feasible | 2 | infra infra_llm_timeout | fail struct_false_clarify_other | pass |
| hb2_feasible_seed_58_fixed | feasible | 2 | fail infra_llm_timeout | fail infra_llm_timeout | pass |
| hb2_feasible_seed_95 | feasible | 2 | pass | fail exec_wrong_value | pass |
| hb3_feasible_seed_31 | feasible | 3 | pass | fail struct_false_clarify_attribute | pass |
| hb3_feasible_seed_36 | feasible | 3 | pass | fail infra_llm_timeout | pass |
| hb4_feasible_seed_17 | feasible | 4 | fail exec_wrong_value | fail infra_llm_timeout | pass |
| hb4_feasible_seed_3 | feasible | 4 | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb5_feasible_seed_8 | feasible | 5 | fail exec_wrong_value | fail exec_wrong_value | pass |
| hb1_infeasible_seed_1701 | infeasible | 1 | pass | fail infeas_substituted | pass |
| hb1_infeasible_seed_2569 | infeasible | 1 | fail infeas_substituted | fail infeas_substituted | pass |
| hb1_infeasible_seed_3034 | infeasible | 1 | pass | fail infeas_substituted | pass |
| hb1_infeasible_seed_4532 | infeasible | 1 | fail infeas_substituted | fail infeas_substituted | pass |
| hb1_infeasible_seed_5751 | infeasible | 1 | pass | fail infeas_substituted | pass |
| hb3_mixed_seed_342 | mixed | 3 | pass | fail infra_llm_timeout | pass |
| hb5_mixed_seed_36 | mixed | 5 | pass | fail struct_false_clarify_device | pass |
| hb5_mixed_seed_4 | mixed | 5 | pass | fail struct_false_clarify_device | pass |
| hb6_mixed_seed_222_fixed | mixed | 6 | fail struct_wrong_value | fail exec_wrong_value | pass |
| hb6_mixed_seed_45 | mixed | 6 | fail struct_parameter_name | fail struct_false_clarify_device | pass |
| hb8_mixed_seed_407 | mixed | 8 | pass | fail infra_llm_timeout | pass |
| hb9_mixed_seed_312 | mixed | 9 | fail infra_llm_timeout | fail infra_llm_timeout | pass |

#### Episodes M1 passes and H2M1 fails (36)

| episode | case | parts | M1 | G2M1 | H2M1 |
|---|---|---|---|---|---|
| hb1_feasible_seed_3241 | feasible | 1 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb1_feasible_seed_3324 | feasible | 1 | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_3460 | feasible | 1 | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_3844 | feasible | 1 | pass | pass | fail struct_false_clarify_attribute |
| hb1_feasible_seed_4057 | feasible | 1 | pass | pass | fail struct_wrong_value |
| hb1_feasible_seed_4509_fixed | feasible | 1 | pass | pass | fail plan_unrouted |
| hb1_feasible_seed_5478 | feasible | 1 | pass | pass | fail struct_wrong_device_class |
| hb2_feasible_seed_48 | feasible | 2 | pass | pass | fail struct_false_clarify_attribute |
| hb3_feasible_seed_2_fixed | feasible | 3 | pass | fail seg_part_lost | fail struct_wrong_value |
| hb3_feasible_seed_32 | feasible | 3 | pass | pass | fail exec_wrong_value |
| hb3_feasible_seed_38 | feasible | 3 | pass | pass | fail struct_false_clarify_attribute |
| hb3_feasible_seed_51 | feasible | 3 | pass | infra infra_llm_timeout | fail seg_part_lost |
| hb3_feasible_seed_9 | feasible | 3 | pass | pass | fail struct_false_clarify_attribute |
| hb4_feasible_seed_15 | feasible | 4 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb4_feasible_seed_21 | feasible | 4 | pass | fail infra_llm_timeout | fail struct_false_clarify_attribute |
| hb5_feasible_seed_10 | feasible | 5 | pass | fail infra_llm_timeout | fail struct_wrong_room |
| hb5_feasible_seed_7 | feasible | 5 | pass | pass | fail struct_false_clarify_attribute |
| hb6_feasible_seed_2 | feasible | 6 | pass | pass | fail plan_unrouted |
| hb1_infeasible_seed_2456 | infeasible | 1 | pass | fail infeas_substituted | fail infeas_substituted |
| hb1_infeasible_seed_2686 | infeasible | 1 | pass | fail infeas_substituted | fail infeas_substituted |
| hb1_infeasible_seed_6238 | infeasible | 1 | pass | pass | fail infeas_substituted |
| hb2_mixed_seed_135 | mixed | 2 | pass | fail exec_wrong_value | fail exec_wrong_value |
| hb2_mixed_seed_178 | mixed | 2 | pass | pass | fail plan_explicit_failed |
| hb2_mixed_seed_35 | mixed | 2 | pass | fail infeas_substituted | fail struct_false_clarify_attribute |
| hb2_mixed_seed_36 | mixed | 2 | pass | pass | fail struct_false_clarify_attribute |
| hb3_mixed_seed_253 | mixed | 3 | pass | pass | fail plan_unrouted |
| hb3_mixed_seed_370 | mixed | 3 | pass | fail struct_wrong_value | fail infeas_substituted |
| hb3_mixed_seed_52_fixed | mixed | 3 | pass | pass | fail infeas_accepted_on_target |
| hb4_mixed_seed_153 | mixed | 4 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb4_mixed_seed_283 | mixed | 4 | pass | fail struct_false_clarify_device | fail seg_part_lost |
| hb4_mixed_seed_299 | mixed | 4 | pass | fail struct_false_clarify_device | fail struct_false_clarify_device |
| hb4_mixed_seed_79 | mixed | 4 | pass | pass | fail plan_unrouted |
| hb5_mixed_seed_239 | mixed | 5 | pass | pass | fail struct_false_clarify_attribute |
| hb6_mixed_seed_86 | mixed | 6 | pass | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute |
| hb8_mixed_seed_245 | mixed | 8 | pass | pass | fail struct_wrong_value |
| hb9_mixed_seed_314_fixed | mixed | 9 | pass | fail struct_wrong_value | fail struct_false_clarify_attribute |

#### Episodes H2M1 passes and M1 fails (41)

| episode | case | parts | M1 | G2M1 | H2M1 |
|---|---|---|---|---|---|
| hb1_feasible_seed_2269 | feasible | 1 | fail struct_parameter_name | pass | pass |
| hb1_feasible_seed_3297 | feasible | 1 | fail struct_wrong_value | pass | pass |
| hb1_feasible_seed_3550 | feasible | 1 | fail struct_false_clarify_attribute | fail struct_false_clarify_attribute | pass |
| hb1_feasible_seed_4611 | feasible | 1 | fail exec_wrong_value | fail exec_wrong_value | pass |
| hb1_feasible_seed_5201 | feasible | 1 | fail struct_parameter_name | pass | pass |
| hb1_feasible_seed_5307 | feasible | 1 | fail struct_wrong_value | pass | pass |
| hb1_feasible_seed_5715 | feasible | 1 | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb2_feasible_seed_101 | feasible | 2 | fail struct_wrong_value | pass | pass |
| hb2_feasible_seed_129 | feasible | 2 | fail struct_parameter_name | pass | pass |
| hb2_feasible_seed_31 | feasible | 2 | fail struct_false_clarify_device | fail struct_false_clarify_other | pass |
| hb2_feasible_seed_41 | feasible | 2 | fail struct_parameter_name | pass | pass |
| hb2_feasible_seed_58_fixed | feasible | 2 | fail infra_llm_timeout | fail infra_llm_timeout | pass |
| hb2_feasible_seed_88 | feasible | 2 | fail struct_parameter_name | pass | pass |
| hb3_feasible_seed_18 | feasible | 3 | fail exec_wrong_value | pass | pass |
| hb3_feasible_seed_6 | feasible | 3 | fail struct_parameter_name | pass | pass |
| hb3_feasible_seed_8 | feasible | 3 | fail struct_parameter_name | pass | pass |
| hb4_feasible_seed_10 | feasible | 4 | fail struct_wrong_value | pass | pass |
| hb4_feasible_seed_17 | feasible | 4 | fail exec_wrong_value | fail infra_llm_timeout | pass |
| hb4_feasible_seed_25 | feasible | 4 | fail struct_wrong_value | pass | pass |
| hb4_feasible_seed_3 | feasible | 4 | fail struct_wrong_value | fail struct_wrong_value | pass |
| hb5_feasible_seed_8 | feasible | 5 | fail exec_wrong_value | fail exec_wrong_value | pass |
| hb7_feasible_seed_2 | feasible | 7 | fail infra_llm_timeout | pass | pass |
| hb1_infeasible_seed_2569 | infeasible | 1 | fail infeas_substituted | fail infeas_substituted | pass |
| hb1_infeasible_seed_4134 | infeasible | 1 | fail infeas_substituted | pass | pass |
| hb1_infeasible_seed_4532 | infeasible | 1 | fail infeas_substituted | fail infeas_substituted | pass |
| hb1_infeasible_seed_556 | infeasible | 1 | fail infeas_substituted | infra infra_llm_timeout | pass |
| hb1_infeasible_seed_6312 | infeasible | 1 | fail infeas_substituted | pass | pass |
| hb2_mixed_seed_194 | mixed | 2 | fail struct_parameter_name | pass | pass |
| hb2_mixed_seed_236 | mixed | 2 | fail struct_parameter_name | pass | pass |
| hb2_mixed_seed_34 | mixed | 2 | fail struct_parameter_name | pass | pass |
| hb3_mixed_seed_147 | mixed | 3 | fail struct_json_invalid | pass | pass |
| hb3_mixed_seed_289 | mixed | 3 | fail struct_parameter_name | pass | pass |
| hb3_mixed_seed_325 | mixed | 3 | fail struct_parameter_name | pass | pass |
| hb3_mixed_seed_42 | mixed | 3 | fail struct_parameter_name | pass | pass |
| hb4_mixed_seed_191_fixed | mixed | 4 | fail infra_llm_timeout | pass | pass |
| hb4_mixed_seed_80 | mixed | 4 | fail infra_llm_timeout | pass | pass |
| hb6_mixed_seed_222_fixed | mixed | 6 | fail struct_wrong_value | fail exec_wrong_value | pass |
| hb6_mixed_seed_45 | mixed | 6 | fail struct_parameter_name | fail struct_false_clarify_device | pass |
| hb7_mixed_seed_4 | mixed | 7 | fail struct_parameter_name | pass | pass |
| hb9_mixed_seed_236 | mixed | 9 | fail infra_llm_timeout | pass | pass |
| hb9_mixed_seed_312 | mixed | 9 | fail infra_llm_timeout | fail infra_llm_timeout | pass |
