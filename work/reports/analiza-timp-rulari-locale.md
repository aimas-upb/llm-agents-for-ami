# Unde se duce timpul în rulările locale HomeBench

Raport de analiză pentru Alex Sorici. Rulări parțiale ECAI-400 pe modele locale (Ollama), 31.10.2026.

Sursa: înregistrările pe episod din `tests/homebench/results/harness/` (M1, M3, qwen2.5-coder-7b, qwen3-8b), logul serverului Ollama și codul din `ami_agents/` și `tests/simuhome/bench/`. Toate cifrele se regenerează cu `python -I work/reports/timing_analysis.py`.

## Rezumat

1. **60% din timpul M1 nu ține de modele.** Două episoade (`hb1_feasible_seed_5275`, `hb1_feasible_seed_4383`) au rămas 2.5 h și 2 h în etapa `execution`, cu eroarea `POST /simulation/fast_forward_to -> URLError WinError 10013`. Împreună fac 4.5 h din cele 7.45 h. Fără ele, M1 are o medie de 78 s/episod (mediana 52 s), adică 400 de episoade ar dura ~8.7 h, nu „nefezabil".
2. **Printre episoadele fără hang, cel mai mare cost sunt generările necontrolate la structuring.** În M1, 49 din 270 de apeluri de structuring (18%) au fost tăiate la timeout-ul de 180 s. Serverul arată că în 47 din 49 de cazuri modelul era încă în curs de generare, după ~8800 de tokeni în median, la un ritm normal de 54 tok/s. Episoadele cu astfel de apeluri au consumat 1.16 h, iar apelurile de structuring în general 89% din timpul episoadelor fără hang.
3. **Cauza probabilă a timeout-urilor este segmentarea cu DS-R1-1.5B, nu încărcarea modelelor.** Segmenterul de 1.5B aproape nu emite eticheta de tip (`achievement` sau `maintenance`). Din 270 de intenturi, 94% au calificatori goi sau doar cu specificitate (`["explicit"]`), iar `route_goal` îi trimite pe ruta „dependency", cu prompt mai lung și cu proprietăți plus mediu în context. În total, 269 din 270 au ajuns pe ruta „dependency". Toate cele 49 de timeout-uri sunt pe această rută. Cu Qwen3-8B ca segmenter, 49 din 49 de intenturi merg pe ruta „simple" și niciunul nu a depășit timeout-ul.
4. **Schimbarea modelelor în VRAM e neglijabilă.** Logul Ollama arată 260 de încărcări în M1 (1.88 pe episod), de 1.0-2.4 s fiecare, în total 7 minute, adică 1.6% din timp. Prefill-ul promptului de sistem e un cost mic, iar logul arată că prefix cache-ul funcționează deja.
5. **De ce pare mai rapid Qwen3-8B singur:** pe cele 12 episoade comune în care M1 și M3 nu au avut timeout-uri, M1 face 59.3 s/episod, iar Qwen3-8B singur 58.1 s. Deci, în același regim, sunt la fel de rapide. Diferența din mediile brute (194 s față de 57 s) vine din cele două hang-uri ale simulatorului și din timeout-urile cauzate de rutarea greșită, nu din numărul de modele sau din thinking.

## 1. Date și limite

| Rulare | Episoade terminate | Interval (31.10) | Segmenter | Structuring |
|---|---|---|---|---|
| M1 | 138 | 00:48-08:18 | DS-R1-1.5B | Qwen3-8B |
| M3 | 21 | 09:08-09:26 | DS-R1-1.5B | DS-R1-7B |
| coder-7b | 50 | 08:25-08:46 | Qwen2.5-Coder-7B | Qwen2.5-Coder-7B |
| qwen3-8b | 20 | 08:47-09:07 | Qwen3-8B | Qwen3-8B |

- Rulările au fost secvențiale pe aceeași mașină (RTX 4070 SUPER, 12 GB), fără suprapunere în timp. Rulările `gpt-5-*` nu intră în analiză.
- Rulările sunt parțiale și au dimensiuni diferite. Comparațiile cinstite se fac pe cele **20 de episoade pe care le-au terminat toate** (primele 20 din listă, 10 din ele de tip `hb1/feasible`) sau pe subsetul de 12 fără timeout.
- Serverul Ollama rulează cu `OLLAMA_NUM_PARALLEL=1`, `FLASH_ATTENTION=true`, `KV_CACHE_TYPE=q8_0`, `KEEP_ALIVE=5m`, iar modelele `*-16k` au `num_ctx=16384` fixat în Modelfile. Cu `NUM_PARALLEL=1`, cererile concurente ale unui episod se pun la coadă în server.
- Secundele pe apel din `meter.calls` includ așteptarea la coadă și, în apelurile tăiate, durata completă până la timeout.
- Nu am analizat acuratețea, doar timpul.

## 2. Unde a mers timpul M1

| Parte | Ore | Pondere |
|---|---|---|
| Hang simulator (etapa `execution` din 2 episoade) | 4.48 | 60.1% |
| Structuring, episoade fără apel tăiat | 1.46 | 19.6% |
| Structuring, episoade cu cel puțin un apel tăiat | 1.16 | 15.5% |
| Planning, profiling, guarding, response, altele | 0.17 | 2.3% |
| Segmentation | 0.13 | 1.8% |
| Netrecut în etape (harness, discovery) | 0.05 | 0.6% |
| **Total** | **7.45** | 100% |

Fără cele două hang-uri: 136 de episoade, 2.95 h, medie 78 s, mediană 52 s. Structuring reprezintă 2.62 h din cele 2.95 h (89%).

Timp pe etapă, secunde pe episod, toate episoadele terminate:

| Rulare | segmentation | structuring | planning | execution | other | total |
|---|---|---|---|---|---|---|
| M1 | 3.5 | 68.2 | 0.7 | 116.8 | 3.6 | 194.3 |
| M3 | 2.4 | 46.8 | 0.1 | 0.0 | 0.0 | 50.7 |
| coder-7b | 6.2 | 14.8 | 0.4 | 0.1 | 1.2 | 24.0 |
| qwen3-8b | 8.7 | 40.7 | 2.1 | 0.1 | 3.7 | 56.9 |

Cele 116.8 s de `execution` din M1 sunt cele două episoade blocate, repartizate pe toate. În restul rulărilor nu a apărut niciun hang, dar M1 are 138 de episoade, iar celelalte 20-50. Cu un hang la ~1.5% din episoade, probabilitatea de a nu vedea niciunul în 20 de episoade e ~74%.

## 3. De ce pare mai rapid Qwen3-8B singur

### 3.1 Pe aceleași episoade

Cele 20 de episoade comune tuturor rulărilor:

| Rulare | s/episod (medie) | s/episod (mediană) | apeluri structuring | tăiate la 180 s | tokeni ieșire/episod (apeluri ok) |
|---|---|---|---|---|---|
| M1 | 86.0 | 64.8 | 32 | 5 | 1876 |
| M3 | 52.0 | 19.3 | 34 | 5 | 834 |
| coder-7b | 19.7 | 12.5 | 49 | 0 | 490 |
| qwen3-8b | 56.9 | 55.5 | 49 | 0 | 2712 |

Aici M1 pare mai lent (86 s față de 57 s), dar 12.3 min din cele 28.7 min ale lui M1 sunt episoade cu un apel tăiat. Dacă le scoatem pe cele cu timeout în M1 sau M3, rămân 12 episoade:

| Rulare | s/episod (medie) | s/episod (mediană) | segmentation | structuring |
|---|---|---|---|---|
| M1 | 59.3 | 41.7 | 3.4 | 48.8 |
| M3 | 20.5 | 18.2 | 2.4 | 16.6 |
| coder-7b | 21.8 | 16.4 | 5.8 | 12.8 |
| qwen3-8b | 58.1 | 55.5 | 9.7 | 42.2 |

M1 și Qwen3-8B singur ajung la același timp pe episod. Segmentarea lui M1 e cu ~6 s mai rapidă (DS-R1-1.5B față de Qwen3-8B care gândește), dar structuring-ul e cu ~6.6 s mai lent (prompt mai lung), deci se anulează. Eșantionul e mic (12 episoade), deci e un indiciu, nu o dovadă.

### 3.2 „Doar unul are thinking" nu descrie M1

În M1 gândesc două modele: DS-R1-1.5B la segmentation și guarding, Qwen3-8B la structuring și profiling. Doar Coder-7B nu gândește. Thinking-ul „la toți pașii" din Qwen3-8B singur adaugă la segmentation ~5 s pe episod față de DS-R1-1.5B (8.7 s față de 3.5 s). Costul real e la structuring, care în ambele configurații e făcut de același Qwen3-8B cu thinking: ~1400 de tokeni de ieșire pe apel și ~54 tok/s la context lung.

### 3.3 Diferența reală: ruta de structuring

`route_goal` (`ami_agents/agents/user_assistant/utils/goal_routing.py`) alege promptul de structuring doar din calificatorii segmenterului: ruta „simple" (prompt de ~12k caractere, fără proprietăți) dacă există exact o etichetă de specificitate, tipul `achievement` și nicio dependență. Altfel, ruta „dependency" (prompt de ~25k caractere, plus proprietăți și variabile de mediu). Codul documentează asta: calificatori lipsă sau contradictorii merg pe „dependency", care e superset, „a mislabelled goal loses nothing but tokens".

Rezultatul, pe toate episoadele terminate:

| Rulare | intenturi | rută „simple" | „dependency": doar specificitate | „dependency": calificatori goi | „dependency": etichetă de dependență |
|---|---|---|---|---|---|
| M1 (seg. DS-R1-1.5B) | 270 | 1 (0%) | 180 (67%) | 73 (27%) | 2 (1%) |
| M3 (seg. DS-R1-1.5B) | 37 | 0 (0%) | 29 (78%) | 8 (22%) | 0 |
| coder-7b | 159 | 48 (30%) | 0 | 0 | 111 (70%) |
| qwen3-8b | 49 | 49 (100%) | 0 | 0 | 0 |

Costul pe rută (apeluri de structuring):

| Rulare | rută | apeluri | tăiate | rată | s (apeluri ok) | tokeni intrare | tokeni ieșire |
|---|---|---|---|---|---|---|---|
| M1 | dependency | 269 | 49 | 18% | 48.9 | 9845 | 1187 |
| M3 | dependency | 37 | 5 | 14% | 14.1 | 9295 | 562 |
| coder-7b | dependency | 111 | 1 | 1% | 16.5 | 9122 | 201 |
| coder-7b | simple | 48 | 0 | 0% | 12.2 | 4584 | 189 |
| qwen3-8b | simple | 49 | 0 | 0% | 36.8 | 4525 | 1107 |

Exemplu concret, aceeași interogare și același text de intent (`hb1_feasible_seed_3460`, „Increase the intensity of the aromatherapy in the living room by 44%"):

| Rulare | Calificatori de la segmenter | Structuring |
|---|---|---|
| M1 | `[]` | timeout la 180 s |
| qwen3-8b | `["explicit", "achievement"]` | 13 s |

Pe cele 20 de episoade comune, Qwen3-8B ca structurator a avut 5 timeout-uri din 32 de apeluri în M1 și 0 din 49 în rularea singură (Fisher exact, p = 0.008).

Legătura e între calificatori și ruta de structuring, iar ruta explică atât prompturile mai lungi (9.8k față de 4.5k tokeni), cât și timeout-urile. Nu am făcut o intervenție care să dovedească cauzalitatea (de exemplu, rularea acelorași episoade cu calificatori corecți forțați). Detalii în secțiunea 7.

## 4. Timeout-uri: generări necontrolate, nu lentoare

- Cele 49 de apeluri eșuate din M1 sunt exact cele 49 de răspunsuri HTTP 500 din logul Ollama, toate cu durata `3m0s`.
- Înainte de fiecare tăiere, serverul genera încă: median 8827 de tokeni, la un ritm median de 54 tok/s. 47 din 49 au depășit 5000 de tokeni. Modelul nu era lent, ci nu se oprea.
- Timeout-ul este `REQUEST_TIMEOUT = 180` în `tests/simuhome/bench/models.py`. Apelurile nu au `max_tokens` pentru modelele locale.
- Cu `NUM_PARALLEL=1`, un apel care generează necontrolat blochează coada pentru restul intenturilor din episod. În M1, episoadele cu 8-9 apeluri de structuring au avut 29 de timeout-uri din 77 de apeluri. Qwen3-8B singur a terminat un episod cu 10 intenturi în 158 s, deci chiar fără rutarea greșită, episoadele hb8-hb10 sunt aproape de limita de 180 s.

Apeluri pe episod și timeout-uri în M1:

| Apeluri/episod | Episoade | Apeluri | Tăiate |
|---|---|---|---|
| 1 | 90 | 90 | 14 |
| 2-3 | 31 | 74 | 6 |
| 4-6 | 6 | 29 | 0 |
| 8-9 | 9 | 77 | 29 |

## 5. Ce nu costă mult

### 5.1 Schimbarea modelelor în VRAM

Din logul Ollama, intervalul de la pornirea runner-ului la „loaded runners":

| Rulare | încărcări | pe episod | durată medie | total | pondere din timp |
|---|---|---|---|---|---|
| M1 | 260 | 1.88 | DS-R1-1.5B 1.0 s, Qwen3-8B 1.8 s, Coder-7B 2.4 s | 7.0 min | 1.6% |
| M3 | 3 | 0.14 | 1.1-2.8 s | 0.1 min | 0.5% |
| coder-7b | 1 | 0.02 | 4.3 s | 0.1 min | 0.4% |
| qwen3-8b | 1 | 0.05 | 2.8 s | 0.0 min | 0.2% |

Intervalul măsurat nu include evacuarea modelului vechi și overhead-ul primului apel. Ca limită superioară, etapele nelegate de structuring (segmentation, planning, response, altele) însumează în M1 ~8 s pe episod, iar în ele intră și munca LLM utilă. Schimbarea modelelor costă deci între ~3 și ~8 s pe episod, sub 10% din cele 78 s. Nu s-a detectat nicio încărcare cu offload parțial pe CPU.

### 5.2 KV caching pentru system prompt

Logul arată că runner-ul (llama-server) refolosește prefixul (`cached n_tokens = 512`, cache de prompturi activ), iar un test separat cu promptul de segmentation (4258 de tokeni) a dat ~0.8-1.0 s la prefill rece și ~40 ms la cald, pe DS-R1-7B. La structuring pe ruta „dependency" intră ~9.8k tokeni, ceea ce înseamnă câteva secunde de prefill dintr-un apel de 30-50 s. Câștigul posibil din cache e deci de ordinul a ~10% din timpul unui apel, mai puțin cât costă generarea. Cache-ul nu se păstrează peste evacuarea modelului, dar la ~1.9 încărcări pe episod asta contează puțin.

## 6. Hang-ul simulatorului

Ambele episoade au `outcome: error`, `error_step: execution`, iar eroarea e:

`SimuHomeError: POST /simulation/fast_forward_to -> URLError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`

Rularea a fost configurată cu `exec_timeout: 60.0`, dar etapa `execution` a durat 8883 s și 7236 s, deci acest timeout nu a limitat-o. De ce nu s-a oprit mai devreme nu am investigat. Ce se vede este că pasul `fast_forward_to` din bucla de ticks (`tests/simuhome/bench/execution.py`) a fost cel care a eșuat, după ore. WinError 10013 înseamnă de obicei o problemă la nivelul sistemului de operare (port blocat sau exclus, firewall); cauza efectivă nu e stabilită.

## 7. Recomandări, în ordinea impactului

| # | Acțiune | Câștig estimat | Observații |
|---|---|---|---|
| 1 | Limită de timp pentru întreaga etapă `execution` și investigarea WinError 10013 la `fast_forward_to`; apoi re-rularea celor două episoade | Până la 4.5 h din 7.45 h în M1; la 400 de episoade, ~8.7 h în loc de peste 20 h | Nu schimbă niciun rezultat al evaluării, doar evită blocajul |
| 2 | Calificatori corecți de la segmenter: alt model pentru segmentation (Qwen3-8B îi dă corect), ieșire constrânsă (schemă JSON) pentru calificatori, sau tratarea absenței tipului ca `achievement` în `route_goal` când există o specificitate | Până la 1.16 h din 2.95 h (39%) din episoadele fără hang, plus promptul mediu de 2× mai scurt | A treia variantă schimbă sistemul evaluat, deci e o decizie de protocol. Ar influența și acuratețea, care nu a fost analizată aici |
| 3 | `max_tokens` pentru apelurile locale de structuring, ca o buclă să se termine în zeci de secunde, nu în 180 s | Reduce costul unui apel necontrolat de la 180 s la câteva zeci | Decizie de protocol (apelurile fără limită sunt în prezent parte din setup). Ghidul modelelor Qwen recomandă să nu se folosească decodare greedy cu thinking, nu am verificat sursa și nu am testat efectul aici |
| 4 | `REQUEST_TIMEOUT` mai mare sau cereri secvențiale pentru episoadele cu 8-10 intenturi, cât timp serverul are `NUM_PARALLEL=1` | Evită timeout-uri provocate doar de coadă | Qwen3-8B singur e la 158 s pe un episod cu 10 intenturi |
| 5 | Swap-uri între modele, KV caching | Mici (câteva secunde pe episod) | Merită doar după 1-4 |

Ce nu am făcut și ar întări concluzia de la secțiunea 3: rularea câtorva episoade M1 cu calificatori corecți forțați la ieșirea segmenterului, pentru a vedea dacă timeout-urile dispar. O repetare a celor 5 episoade cu timeout din M1 ar arăta și cât de reproductibile sunt (decodarea e greedy, dar rezultatul numeric poate varia).

## 8. Reproducere și limite

```
python -I work/reports/timing_analysis.py
```

Scriptul citește rezultatele din `tests/homebench/results/harness/` (în `.gitignore`) și logul `%LOCALAPPDATA%\Ollama\server.log`. Cifrele se vor schimba pe măsură ce rulările avansează.

Limite:

- Rulări parțiale, cu eșantioane mici: 20 de episoade comune și 12 fără timeout. Episoadele comune sunt primele din listă și au mulți `hb1`.
- Variantele M2 și M4-M8 nu au fost rulate local și nu sunt analizate.
- Tokenii și secundele pe apel vin din `meter` al harness-ului, nu din server. Secundele includ așteptarea la coadă.
- Rutarea explică prompturile mai lungi și apare doar în apelurile tăiate, dar legătura cauzală completă (calificatori goi duc la ruta grea, care duce la generări necontrolate) nu a fost testată prin intervenție.
- Nu s-a analizat efectul rutării asupra corectitudinii goal-urilor structurate.
