# Integrarea benchmark-ului SimuHome în cadrul AmI HMAS

**Raport tehnic**

---

## Rezumat executiv

- Scenariile benchmark-ului SimuHome sunt expuse acum ca **Thing Description-uri W3C WoT** într-o platformă HMAS, ceea ce permite evaluarea directă a agenților AmI pe acest benchmark, cu rezultate comparabile cu cele publicate. **[contribuție originală]**

- Conversia se face **direct din simulator în TD**, fără intermediarul Home Assistant. Traseul prin HA pierdea în tăcere aproximativ o cincime din dispozitivele corpusului și aplatiza semantica protocolului Matter, în schimbul unei refolosiri de cod care oricum trebuia rescrisă.

- Toate faptele despre protocolul Matter provin dintr-un registru vendorizat și fixat pe o versiune anume a specificației, niciodată din memoria modelului de limbaj. Această disciplină s-a dovedit necesară, nu doar prudentă.

- Reprezentarea este **îmbogățită semantic prin SOSA și TD-SOSA**: fiecare dispozitiv declară explicit ce percepe din mediu și ce poate influența în el. Fără acest strat, cererile implicite — în care utilizatorul descrie un disconfort, nu o comandă — nu ar putea fi tratate decât pe baza cunoștințelor implicite ale modelului de limbaj. **[contribuție originală]**

- A fost dezvoltată ontologia **HomeOnt**, care consolidează vocabularul necesar pentru spațiile locuinței, familiile de dispozitive și adnotările proiecției TD, acolo unde vocabularele standard nu acoperă domeniul. **[contribuție originală]**

- Camera este modelată ca două resurse distincte — spațiul fizic și mediul din el. Un congelator se află într-o bucătărie, dar nu observă aerul acesteia, iar reprezentarea trebuie să spună asta.

- Infrastructura permite **rularea paralelă izolată**: fiecare worker are propriul simulator și propria proiecție TD, cu ceasuri și dinamici de mediu complet independente.

- Scorarea folosește **evaluatorul propriu al SimuHome**, ceea ce menține comparabilitatea rezultatelor. Lanțul complet a fost validat: acțiuni trimise prin Thing Description-uri satisfac obiectivele unui episod real.

- Fazele de conversie, invocare, notificare și îmbogățire semantică sunt finalizate și acoperite de suite de verificare automată; harness-ul de rulare în paralel este în lucru.

---

## 1. Introducere: simulatorul SimuHome

### 1.1 Rolul simulatorului

SimuHome este un simulator de locuință inteligentă construit pentru evaluarea agenților conversaționali în medii ambientale. Ceea ce îl deosebește de un mediu de test static este faptul că rulează un fir de simulare continuu: la fiecare *tick* — interval implicit de 0,1 secunde — recalculează starea mediului fiecărei camere, astfel încât temperatura, umiditatea, iluminarea și concentrația de particule evoluează în timp chiar și atunci când agentul nu întreprinde nimic.

Proprietatea aceasta are o consecință care se propagă în toată arhitectura descrisă mai jos. Un plan care presupune tacit că lumea rămâne neschimbată între două acțiuni consecutive este un plan construit pe o premisă falsă, iar simulatorul îl va invalida. Prin construcție, deci, SimuHome obligă agentul să verifice starea mediului în loc să o presupună — o cerință care a determinat, printre altele, decizia de a nu modela timpi de stabilizare în reprezentarea TD, ci de a lăsa planurile să confirme rezultatul printr-un nod de verificare.

### 1.2 Protocolul Matter

Dispozitivele sunt expuse nativ în Matter (Connectivity Standards Alliance, 2025), standardul industrial pentru interoperabilitatea echipamentelor de casă inteligentă. Modelul său de date este ierarhic: un dispozitiv are unul sau mai multe *endpoint*-uri, fiecare endpoint grupează funcționalitatea în *clustere* (`OnOff`, `LevelControl`, `Thermostat`), iar fiecare cluster expune *atribute* care poartă starea și *comenzi* care o modifică.

Există aici o distincție de care depinde întreaga arhitectură a conversiei, și pe care este ușor să o ratezi: a fi scriptibil nu înseamnă a fi acționabil. În Matter, acționarea se face pe două căi — prin scrierea directă a unui atribut, dar și prin invocarea unei comenzi de cluster. Exemplul cel mai clar este chiar cel mai frecvent atribut din corpus: `OnOff.OnOff` este declarat *read-only* în specificație, ceea ce ar sugera că nu poate fi modificat, însă el este condus de comenzile `On`, `Off` și `Toggle` și rămâne atributul cel mai acționat din întregul benchmark. O conversie care ar deduce acționabilitatea doar din permisiunea de scriere ar pierde tocmai capabilitățile cele mai importante.

### 1.3 Dispozitivele din corpus

Corpusul cuprinde șase sute de episoade, însumând peste șaisprezece mii de instanțe de dispozitive distribuite în peste două sute de configurații distincte de locuință. Diversitatea aceasta nu este ornamentală: fiecare seed descrie efectiv o altă casă, ceea ce a impus ca identificatorul locuinței să facă parte din schema de identitate a reprezentației.

Cele șaisprezece tipuri de dispozitive acoperă iluminatul (becuri simple și cu reglaj de intensitate), climatizarea (aparate de aer condiționat, pompe de căldură, ventilatoare), tratarea aerului (purificatoare, umidificatoare, dezumidificatoare), umbrirea (jaluzele motorizate) și electrocasnicele obișnuite — mașini de spălat și uscat rufe, mașini de spălat vase, frigidere, congelatoare, aspiratoare robot și televizoare.

Structura spațială este remarcabil de regulată. Fiecare episod are exact cinci camere, iar fiecare dispozitiv aparține exact unei camere — o partiție curată, verificată pe toate instanțele corpusului, fără nicio excepție. La nivelul întregului benchmark apar nouă identificatori de cameră, dintre care patru sunt prezenți în toate episoadele, iar al cincilea variază.

O observație care va deveni importantă în secțiunea dedicată modelării semantice: SimuHome nu conține niciun senzor autonom. Toate cele șaisprezece tipuri sunt aparate sau actuatoare. Percepția mediului există totuși, dar este încorporată în aparate — iar felul în care este încorporată nu este uniform, ceea ce a necesitat o analiză separată.

### 1.4 Variabilele de mediu

Simulatorul urmărește patru variabile pentru fiecare cameră: temperatura aerului, umiditatea relativă, iluminarea și concentrația de particule PM10. Primele două sunt stocate în centi-unități, conform convenției Matter — o valoare de 2362 înseamnă 23,62 °C — ceea ce a impus o atenție specială la conversie, întrucât o proiecție care ar raporta valoarea brută în timp ce declară gradul Celsius ca unitate ar induce în eroare orice consumator.

Simulatorul își expune propriul model cauzal la un endpoint dedicat, indicând pentru fiecare variabilă de mediu ce tipuri de dispozitive o pot influența și prin ce acțiuni. Acest endpoint a devenit, în arhitectura noastră, sursa de adevăr pentru afirmațiile de efect — o decizie discutată pe larg în secțiunea 5.

---

## 2. Tipurile de cereri din SimuHome și evaluarea lor

### 2.1 Taxonomia cererilor

Benchmark-ul distribuie uniform cele șase sute de episoade pe șase tipuri de interogare, echilibrate în mod egal între cazuri fezabile și nefezabile. Diferența dintre aceste tipuri nu este de dificultate lexicală, ci de cât de mult trebuie să deducă agentul.

La un capăt al spectrului se află **cererile explicite** (qt3), în care utilizatorul numește dispozitivul, modul de funcționare și parametrii: *„Turn on the living room AC 1, switch it to cooling mode and set the fan to sixty percent."* Aici agentul are de tradus o instrucțiune, nu de interpretat o intenție.

**Interogările de stare** (qt1) cer o informație în loc de o acțiune — *„how is the humidity in here right now?"* — și testează capacitatea agentului de a localiza sursa corectă a unei valori.

**Cererile cu constrângeri temporale** (qt4, cu trei subtipuri) introduc programarea relativă: *„23 minutes before the washer 1 in the utility room finishes, set refrigerator 2 in the kitchen to 6.0 °C."* Ele presupun raportarea la un eveniment viitor din interiorul simulării.

La celălalt capăt se află **cererile implicite** (qt2), care sunt și motivul principal pentru investiția semantică descrisă în acest raport. Utilizatorul descrie un disconfort, fără a numi nici dispozitivul, nici acțiunea:

> *„Ugh, the study is really dim, my eyes keep squinting and starting to ache, and the living room is the same, pretty dim, makes me a bit annoyed since I was thinking of flipping through that book there."*

Pentru a răspunde, agentul trebuie să rezolve două întrebări succesive. Întâi, care variabilă de mediu corespunde plângerii — aici, iluminarea. Apoi, ce capabilități pot influența acea variabilă, în acele camere anume. A doua întrebare este cea dificilă: fără o modelare semantică explicită a efectelor, ea nu poate primi răspuns decât din cunoștințele implicite ale modelului de limbaj. Iar dacă răspunsul vine de acolo, atunci benchmark-ul măsoară ce își amintește modelul din pre-antrenare, nu cât de bine planifică într-un mediu dat.

### 2.2 Structura de evaluare a unui episod

Fiecare episod poartă o secțiune de evaluare cu două componente complementare. Prima listează instrumentele pe care agentul trebuia să le invoce — o verificare de proces, care confirmă că agentul a explorat mediul înainte de a acționa. A doua enumeră obiectivele: pentru fiecare cameră vizată, variabila de mediu și direcția în care ea trebuia să se modifice.

```json
"eval": {
  "required_actions": [
    {"tool": "get_room_devices", "params": {"room_id": "study_room"}}
  ],
  "goals": [
    {"room_id": "study_room", "room_state": "illuminance",
     "direction": "increase", "feasibility": true}
  ]
}
```

Scorul final combină ambele componente: un agent care atinge obiectivul fără să fi explorat mediul, sau care explorează corect dar nu obține efectul, nu primește punctajul.

### 2.3 Evaluarea contrafactuală

Cel mai subtil aspect al evaluatorului SimuHome — și cel care a determinat în bună măsură proiectarea infrastructurii noastre de rulare — este că evaluarea nu compară starea finală cu cea inițială, ci cu o linie de bază calculată separat.

Procedura este următoarea: după terminarea episodului se captează starea finală a locuinței, apoi simulatorul este resetat la configurația inițială și avansat rapid, fără niciun agent care să acționeze, până la exact același tick. Abia atunci se compară valoarea finală cu valoarea acestei linii de bază.

Motivul devine evident dacă ne întoarcem la observația din secțiunea 1.1: mediul evoluează singur. Dacă iluminarea unei camere ar fi crescut oricum, din cauza orei simulate, un agent complet pasiv ar apărea drept „reușit" într-o comparație naivă între starea inițială și cea finală. Linia de bază contrafactuală elimină exact acest artefact, izolând contribuția reală a agentului.

Pentru infrastructură, aceasta impune două constrângeri care nu sunt evidente la prima vedere: starea finală trebuie captată **înainte** ca evaluatorul să preia controlul simulatorului, iar evaluatorul are nevoie de simulator în exclusivitate cât timp își calculează linia de bază. Ambele au fost încorporate în modelul de worker descris în secțiunea 7.

---

## 3. Conversia scenariilor SimuHome în reprezentare HMAS/TD

### 3.1 Decizia: de ce direct, și nu prin Home Assistant

Traseul avut inițial în vedere trecea prin Home Assistant: dispozitivele SimuHome urmau să fie exportate în format YAML, importate în HA, iar adaptorul HASP existent — care deservește deja implementarea reală din laboratorul lab308e — le-ar fi expus ca Thing Description-uri. Motivația era refolosirea unei componente funcționale și testate.

Această cale a fost abandonată în urma unor constatări obținute experimental, iar prima dintre ele a fost și cea decisivă. Componenta `virtual` din Home Assistant, folosită pentru a instanția dispozitive fictive, pur și simplu **nu are platformă `climate`**. Un import de probă a arătat că entitățile de tip climate și cover nu sunt create deloc, în timp ce toate celelalte tipuri trec fără probleme. Mai grav, Home Assistant nu semnalează nimic în jurnal: platforma nu figurează în lista celor suportate, iar entitatea este eliminată tăcut la încărcarea configurației. Extrapolat la întregul corpus, aceasta însemna că toate aparatele de aer condiționat, toate pompele de căldură și toate jaluzelele — împreună, aproximativ o cincime din dispozitivele benchmark-ului — ar fi lipsit din mediul pe care agenții urmau să îl exploreze, fără niciun indiciu că lipsesc.

La aceasta se adaugă o degradare semantică sistematică. Structura cluster/comandă a protocolului Matter se aplatizează în domeniile Home Assistant: modurile de funcționare devin numere întregi opace, iar lista de etichete `SupportedModes` — exact informația de care are nevoie un agent pentru a înțelege ce înseamnă „modul Heavy" la o mașină de spălat — depășește limita de 255 de caractere impusă de HA pentru starea unei entități și ajunge stocată ca valoare necunoscută.

Ar fi existat o alternativă, integrarea `virtual_devices` din HACS, care are suport pentru climatizare. Costul ei însă era prohibitiv: se configurează exclusiv prin asistent grafic, cu un singur tip de dispozitiv per intrare de configurare, ceea ce pentru șase sute de episoade a aproape treizeci de dispozitive fiecare ar fi însemnat parcurgerea manuală a mii de fluxuri interactive.

În fine, argumentul care a făcut întreaga discuție să pară retrospectiv inutilă: SimuHome expune deja un server API complet, iar endpoint-ul său de resetare acceptă **verbatim** câmpul de configurare din fișierul de episod. Nu era nevoie de nicio conversie intermediară, de niciun format intermediar și de niciun import.

Bilanțul este, așadar, clar. Intermediarul Home Assistant adăuga pierderea unei cincimi din corpus și o degradare semantică notabilă, în schimbul unei refolosiri de cod care oricum ar fi trebuit rescrisă pentru a genera payload-uri Matter. Componenta dezvoltată — pe care o vom numi SHTD, *SimuHome Thing Descriptions* — elimină simultan Home Assistant, convertorul YAML și simulatorul auxiliar care compensa lipsurile acestuia.

### 3.2 De ce Thing Description, și de ce cu semantică

Alegerea formatului TD nu ține de preferință, ci decurge din trei cerințe care acționează simultan.

Prima este că agenții AmI consumă exclusiv Thing Description-uri (W3C, 2023). O verificare a stratului de agenți a confirmat că singurul URL codificat în cod este rădăcina platformei; restul navigării se face hipermedia, urmând relațiile din graf — platforma își declară workspace-urile, workspace-urile își declară artefactele, iar artefactele își declară capabilitățile cu ținta de invocare atașată. Identitatea unei capabilități *este* href-ul formularului ei.

A doua cerință este ca descoperirea mediului să fie autonomă. Agentul nu primește o listă de dispozitive pe care să o parcurgă; el explorează graful pornind de la un singur punct de intrare. Un format care ar descrie doar valori, fără relații de containment și fără tipuri, nu ar permite acest lucru.

A treia, și cea care justifică partea semantică, este că planificarea are nevoie de sens, nu doar de sintaxă. Un planificator care vede `SystemMode = 3` nu poate folosi această valoare dacă nu știe că 3 înseamnă *Cool*. Iar dacă acea cunoștință provine din pre-antrenarea modelului de limbaj, ne întoarcem la problema semnalată în secțiunea 2.1: evaluarea ajunge să măsoare memoria modelului. Un TD îmbogățit semantic rezolvă această problemă legând valorile de vocabulare partajate — SOSA și SSN (W3C, 2017a, 2017b) pentru observație și acționare, QUDT (QUDT.org, 2024) pentru mărimi și unități, SAREF (ETSI, 2020a) pentru clasificarea dispozitivelor — astfel încât sensul devine verificabil în loc de presupus.

### 3.3 Registrul Matter vendorizat

Principiul care a ghidat toată conversia se poate formula într-o singură propoziție: niciun fapt despre protocolul Matter — tip de dispozitiv, cluster, atribut, comandă, enumerare — nu provine din memoria modelului de limbaj.

În consecință, a fost vendorizat un registru complet offline, fixat pe o versiune anume din depozitul oficial connectedhomeip (Connectivity Standards Alliance, 2025) (tag `v1.5.1.0`, commit `abcc720b48c5`, specificația 1.5.1). Registrul normalizat cuprinde peste o sută treizeci de clustere, nouăzeci și unu de tipuri de dispozitive și câteva sute de definiții de enumerări și structuri, iar rezoluția lui pe corpus este completă: fiecare cale de atribut întâlnită în cele șase sute de episoade se rezolvă.

Normalizarea a cerut mai multă atenție decât ar sugera formatul XML de intrare, iar un aspect merită menționat pentru că a produs inițial rezultate greșite. Un număr semnificativ de clustere sunt declarate ca derivate dintr-un cluster de bază și își omit atributele moștenite. Complicația este că un cluster derivat re-declară adesea un atribut moștenit doar pentru a-i adăuga o condiție de conformitate, fără a repeta tipul sau permisiunile — `DishwasherMode` redeclară astfel `CurrentMode` fără tip și fără secțiune de acces, pentru că clusterul de bază spune deja că este `uint8`. O fuziune care ar înlocui definiția moștenită cu cea locală ar pierde tocmai informația esențială. Corectarea, prin fuziune câmp cu câmp, a recuperat douăzeci de atribute scriptibile care fuseseră inițial clasificate greșit ca read-only.

### 3.4 Schema de identitate

Containment-ul este exprimat prin sub-căi reale, niciodată prin separatori în interiorul unui singur segment de cale:

```
GET  /                                              platformă
GET  /workspaces/{locuință}                         workspace locuință
GET  /workspaces/{locuință}/{cameră}                workspace cameră
GET  /workspaces/{locuință}/{cameră}/artifacts/{dispozitiv}   Thing Description
GET  .../artifacts/{dispozitiv}/properties/{nume}   citirea unei proprietăți
POST .../artifacts/{dispozitiv}/actions/{nume}      invocarea unei acțiuni
POST /hub/                                          abonare la notificări
```

O decizie explicită merită subliniată: **identitatea Matter rămâne în afara URL-ului**. Segmentul final este numele capabilității așa cum apare în Thing Description — `onOff`, `currentMode`, `temperatureSetpoint` — iar SHTD rezolvă intern clusterul, atributul și endpoint-ul corespunzător. Raționamentul este că numele Matter constituie proveniență, nu identitate, iar un planificator care ajunge să construiască un URL în loc să îl copieze are nevoie de convenția cea mai simplă cu putință. Coordonatele Matter rămân disponibile în corpul TD-ului, deci nimic nu se pierde.

Identificatorul locuinței, la rândul lui, nu este decorativ. Cum s-a arătat în secțiunea 1.3, fiecare seed descrie o casă diferită, cu peste două sute de configurații distincte în corpus, astfel încât fără el IRI-urile a două episoade ar intra în coliziune.

Camerele s-au dovedit stratul intermediar potrivit din două motive convergente: partiția dispozitivelor pe camere este curată, iar camera este entitatea care poartă efectiv variabilele de mediu.

### 3.5 Contracte impuse de crawler-ul agenților

Două constrângeri au fost citite direct din codul agenților, nu presupuse, iar ambele au modificat forma RDF-ului emis. Crawler-ul coboară într-o resursă conținută doar dacă tipul acesteia este declarat **în graful părintelui**, și acceptă o resursă drept artefact doar dacă tipul de artefact apare **în graful workspace-ului**. De aceea documentul locuinței tipizează inline fiecare cameră, iar fiecare document de cameră își tipizează inline dispozitivele.

Din același motiv, relațiile de containment sunt emise în ambele sensuri. Ontologia HMAS (Hyperagents, 2023) le declară drept inverse una alteia, ceea ce ar face redundantă a doua afirmație pentru un raționator — dar un crawler simplu nu face inferență, iar navigarea înapoi în ierarhie trebuie să fie posibilă dintr-un singur document.

### 3.6 Proiecție fără stare

SHTD nu deține nicio stare proprie a lumii. Fiecare cerere recitește starea completă de la simulator. Consecința practică este importantă pentru rularea în serie a episoadelor: resetarea simulatorului schimbă episodul, iar proiecția urmează la următoarea cerere, fără niciun pas de invalidare a vreunui cache. Măsurătorile au confirmat că o resetare durează sub o zecime de secundă și că proiecția a urmărit corect trecerea între locuințe cu numere diferite de camere și dispozitive, fără repornire.

### 3.7 Descrierea valorilor pentru planificator

Schemele descriu forma unei valori, niciodată valoarea însăși. Distincția pare pedantă până când o raportăm la natura continuă a simulării: un document care ar include valoarea curentă ar îngheța, într-o reprezentare cache-abilă, o stare care se schimbă la fiecare tick. Valoarea se obține prin dereferențierea formularului de citire, care este singura cale corectă.

În schimb, două tipuri de descrieri generate automat fac valorile inteligibile pentru un planificator bazat pe modele de limbaj.

Pentru **enumerări**, valoarea transmisă pe fir este întotdeauna întregul, iar numele este doar modul în care planificatorul recunoaște întregul potrivit:

```turtle
td:hasInputSchema [ a js:IntegerSchema ;
    js:enum 0, 1, 3, 4, 5, 6, 7, 8, 9 ;
    js:description "0 = Off (the Thermostat does not generate demand for
      Cooling or Heating). 1 = Auto (...). 3 = Cool (demand is only generated
      for Cooling). 4 = Heat (...). 7 = FanOnly." ] ;
```

Aceasta închide o breșă concretă: regulile proprii ale simulatorului cer `SystemMode = 3` pentru răcire, iar fără descriere agentul ar trebui să deducă din pre-antrenare că 3 înseamnă *Cool*. Textul explicativ este preluat din specificația Matter acolo unde există și spune ceva ce numele nu spune, și este omis acolo unde ar fi doar o reformulare sau o trimitere la o secțiune de document inaccesibilă agentului.

Pentru **limitele per dispozitiv**, alegerea a fost să fie exprimate ca restricții de schemă în vocabularul standard, nu ca text:

```turtle
td:hasOutputSchema [ a js:IntegerSchema ;
    qudt:unit unit:DEG_C ;
    js:minimum 5.0 ;
    js:maximum 95.0 ] ;
```

Corolarul acestei decizii este că atributele care poartă limitele — `MinTemperature`, `MaxLevel`, `Min/MaxMeasuredValue` — nu mai sunt expuse ca proprietăți citibile de sine stătătoare. O limită este o constrângere asupra unei valori, nu o stare separată; publicarea ei în ambele forme ar afirma același lucru de două ori și ar lăsa un validator fără nimic de verificat, întrucât textul liber nu este verificabil automat.

---

## 4. Notificarea schimbărilor de stare

Componenta aceasta nu figurează explicit în structura cerută a raportului, dar merită consemnată pentru că este condiția de corectitudine a întregii arhitecturi: un agent care acționează și apoi planifică pe baza stării pe care și-o amintește planifică într-o lume care nu mai există.

Simulatorul nu poate emite notificări — o verificare a codului său a confirmat absența oricărui websocket, flux de evenimente sau webhook. SHTD interpune deci un mecanism de interogare periodică a stării complete, calculează diferența față de instantaneul anterior și distribuie o notificare pentru fiecare atribut modificat, prin protocolul WebSub (W3C, 2018).

Capabilitățile de abonare sunt declarate pe resursa la care se referă, astfel încât un agent să poată alege granularitatea: abonarea la workspace-ul locuinței aduce notificări de la toate dispozitivele și de la toate mediile, abonarea la o cameră le restrânge la dispozitivele acelei camere și la mediul ei, iar abonarea la un artefact aduce doar schimbările acelui artefact.

O rafinare descoperită la testare merită menționată, pentru că ilustrează genul de eroare pe care doar măsurarea o scoate la iveală. Diferența se calcula inițial pe valoarea brută, în centi-unități. Rezultatul era că o derivă de la 2330 la 2331 — complet invizibilă la afișare, ambele valori fiind 23,3 °C — trezea toți abonații de aproximativ două ori pe secundă. Calculând diferența pe valoarea efectiv raportată, notificările inutile au dispărut complet, în timp ce derivele reale continuă să fie semnalate.

---

## 5. Modelarea SOSA și TD-SOSA: percepții și efecte

### 5.1 Motivația: tratarea cererilor implicite

Pentru o cerere explicită, un TD pur sintactic este suficient: dispozitivul este numit, acțiunea este numită, iar agentul are doar de tradus. Pentru o cerere implicită, în schimb, agentul trebuie să poată interoga mediul semantic, iar cele două întrebări din secțiunea 2.1 devin interogări concrete asupra grafului:

```sparql
# ce poate crește iluminarea acestei camere?
?art td:hasActionAffordance ?a .
?a tdsosa:hasEffectActuation ?act ; td:hasForm/hctl:hasTarget ?href .
?act sosa:actsOnProperty <…/study_room#illuminance> .

# cine îmi poate spune temperatura acestei camere?
?sensor sosa:observes <…/utility_room#air_temperature> .
```

Ambele primesc acum răspuns direct din reprezentare. Fără stratul TD-SOSA, prima ar necesita ca modelul să știe din pre-antrenare că becurile influențează iluminarea — o presupunere plauzibilă în general, dar care nu spune nimic despre *acest* simulator anume. Un contraexemplu concret: în SimuHome ventilatorul **nu** modifică temperatura camerei, deși un raționament de bun-simț ar sugera contrariul. Cunoștințele generale ale modelului ar fi, în acest caz, active înșelătoare.

### 5.2 Percepția: ce observă fiecare dispozitiv

Deși SimuHome nu are senzori autonomi, șase familii de dispozitive poartă clustere de măsurare. Întrebarea relevantă pentru modelare nu este dacă măsoară, ci **ce anume** măsoară — iar răspunsul, obținut prin compararea fiecărei citiri cu starea camerei corespunzătoare pe toate cele șase sute de episoade, s-a dovedit perfect dihotomic, fără nicio excepție în niciunul dintre sensuri.

Aparatele de aer condiționat, pompele de căldură, umidificatoarele și dezumidificatoarele observă efectiv aerul camerei în care se află: pompa de căldură din camera tehnică raportează 23,69 °C într-o cameră aflată la 23,69 °C. Frigiderele și congelatoarele, dimpotrivă, își măsoară propriul compartiment — congelatorul din bucătărie raportează −15,0 °C într-o bucătărie de 23,62 °C.

Distincția nu este una de nuanță. A modela congelatorul ca observator al bucătăriei ar fi pur și simplu **fals**, nu doar redundant, iar un agent care ar căuta temperatura bucătăriei ar primi un răspuns greșit cu peste treizeci și opt de grade. În consecință, primele patru familii sunt tipizate ca senzori și declară ce proprietate a camerei observă, împreună cu relația inversă — direcția pe care o interoghează efectiv planificatorul când caută cine îi poate spune o valoare. Ultimele două primesc un obiect de interes propriu, interiorul aparatului, pentru că observă ceva real, doar că nu camera.

Merită consemnat explicit și un aspect negativ: iluminarea și concentrația de particule nu au niciun senzor în tot corpusul. Pentru aceste două variabile, proprietatea camerei rămâne singura sursă. Aceasta este forma simulatorului, nu o scurtătură de modelare, și este mai bine spusă decât lăsată să fie descoperită.

### 5.3 Efectele: ce poate influența fiecare acțiune

Rândurile aprobate din tabelul de efecte se atașează capabilităților de acțiune ca actuări cu efect declarat:

```turtle
<…/actions/onOff> tdsosa:hasEffectActuation [
    a sosa:Actuation ;
    sosa:actsOnProperty <…/kitchen#illuminance> ;
    tdsosa:affectsObservableProperty <…/kitchen#illuminance> ] .
```

Direcționalitatea urmează o regulă stabilită explicit în urma discuțiilor de proiectare: comenzile de tip pornit/oprit poartă o direcție, în timp ce capabilitățile care transportă o valoare sunt fără direcție, întrucât rezultatul depinde de valoarea trimisă, nu de simplul act al invocării. Cum o capabilitate este indexată pe atribut, iar `onOff(true|false)` conduce toate cele șase comenzi On/Off ale clusterului, o direcție supraviețuiește doar dacă toate comenzile care conduc acel atribut sunt de acord asupra ei. Rezultatul este că o acțiune nu afirmă niciodată simultan că o variabilă crește și că scade — o incoerență care apăruse în prima variantă a generatorului.

Un punct de principiu, discutat și stabilit explicit: **aceste afirmații sunt indicii, nu rețete.** Ele restrâng căutarea planificatorului la comenzile care pot mișca o anumită variabilă, dar nu îi spun cum să le combine. Deducerea faptului că răcirea unei camere necesită `SystemMode = 3` **împreună cu** un setpoint sub temperatura curentă, în ordinea corectă, rămâne sarcina planificatorului BehaviorTree — și este exact capacitatea pe care benchmark-ul o măsoară. A codifica rețeta completă în Thing Description ar însemna a-i oferi planificatorului răspunsul și a înceta să măsurăm tocmai lucrul pentru care benchmark-ul există.

O a doua decizie de principiu privește alinierea la simulator. Afirmațiile de efect au fost restrânse deliberat la ceea ce SimuHome modelează efectiv. Ventilatorul și jaluzeaua au fost eliminate din tabel: deși fizic un ventilator produce o senzație de răcorire, iar o jaluzea reglează evident lumina naturală, simulatorul nu modelează aceste dinamici. O afirmație pe care simulatorul nu o va onora ar face ca un plan să pară corect în timp ce valoarea măsurată nu se mișcă — ceea ce ar invalida orice comparație cu rezultatele publicate ale benchmark-ului. Decizia este consemnată ca reversibilă: dacă simulatorul va fi extins cu aceste dinamici, cele două intrări pot fi restaurate.

O breșă sistematică descoperită târziu ilustrează valoarea verificării automate. Tabelul enumera inițial doar comenzile de cluster, deci scrierile de atribute nu erau candidate la afirmații de efect. Or, regulile proprii ale simulatorului listează mai multe astfel de scrieri ca participanți obligatorii sau opționali la modificarea unei variabile. Cinci din cele șapte familii sub-declarau; cel mai grav, aparatul de aer condiționat nu declara niciun efect pentru `SystemMode` și pentru setpoint-ul de răcire, deși ambele sunt **obligatorii** pentru ca răcirea să se producă. Rândurile lipsă au fost generate din regulile simulatorului, cu proveniența consemnată în tabel.

### 5.4 Reconcilierea cu adevărul simulatorului

Ceea ce afirmă graful este confruntat automat cu modelul cauzal propriu al simulatorului, pentru fiecare dintre cele patru variabile de mediu. Verificarea se face la nivel de **acțiune individuală**, nu doar de tip de dispozitiv — o granularitate grosieră trecea de îndată ce o singură acțiune a unei familii revendica efectul, ceea ce este exact motivul pentru care sub-declararea descrisă mai sus a putut trece neobservată o vreme. Cu verificarea la granularitatea corectă, potrivirea este exactă pe toate cele patru variabile, testată pe episoade cu seturi diferite de dispozitive.

---

## 6. Ontologia HomeOnt

### 6.1 Scop

Vocabularele standard acoperă o parte din necesar, dar niciunul nu acoperă domeniul în întregime. SAREF (ETSI, 2020a) definește exact cinci subclase de dispozitiv — actuator, aparat electrocasnic, echipament HVAC, contor și senzor — utile pentru clasificarea generală, dar insuficiente pentru familiile concrete din corpus. SAREF4BLDG (ETSI, 2020b) oferă noțiunea de spațiu de clădire, însă fără nicio subclasă, deci fără tipuri de încăperi. Brick (Brick Consortium, 2024) are un catalog bogat de tipuri de încăperi, dar orientat spre clădiri comerciale, iar dintre cele nouă tipuri necesare aici doar biroul există. DogOnt (Bonino & Corno, 2008), cea mai apropiată ca domeniu, acoperă cinci din cele nouă.

Cum niciun vocabular existent nu acoperă simultan tipurile de încăperi ale corpusului, familiile de dispozitive și adnotările necesare proiecției TD, a fost dezvoltată ontologia **HomeOnt**, care consolidează acest strat.

### 6.2 Conținut

Ontologia grupează termenii în cinci categorii. Spațiile de clădire formează o ierarhie cu subtipuri (camera copiilor, dormitorul de oaspeți și dormitorul matrimonial fiind specializări ale dormitorului). Mediul unui spațiu este o clasă distinctă, subordonată noțiunii SOSA de obiect de interes, legată de spațiu prin relații reciproce. Situarea artefactelor este exprimată prin relații spațiale proprii, separate de containment-ul logic al workspace-urilor HMAS. Familiile de dispozitive sunt ancorate în SAREF. În fine, proprietățile de mediu poartă mărimea fizică și unitatea QUDT corespunzătoare, iar un mic set de adnotări deservește proiecția TD.

### 6.3 Separarea loc / mediu

Cea mai importantă decizie de modelare din ontologie, și cea care a corectat o eroare conceptuală din prima variantă, este că o cameră este **două resurse distincte**, nu una.

Există, pe de o parte, spațiul fizic — o încăpere de tipul bucătărie, un spațiu de clădire — în care dispozitivele se află. Există, pe de altă parte, mediul din acel spațiu, care este obiectul de interes ale cărui proprietăți observabile sunt măsurate și modificate. **Fiecare** dispozitiv este situat în spațiu; **doar** dispozitivele care percep sau influențează efectiv o variabilă a mediului se leagă de mediu.

Confundarea celor două ar fi însemnat ca orice dispozitiv aflat într-o cameră să apară drept observator al aerului acelei camere. Congelatorul din secțiunea 5.2 demonstrează concret de ce aceasta ar fi fost o afirmație factual greșită, nu doar o redundanță inofensivă.

### 6.4 Principii de proiectare

Trei principii au ghidat conținutul ontologiei, iar respectarea lor a dus în mai multe rânduri la **eliminarea** unor termeni introduși anterior.

Primul: nu se inventează nimic acolo unde există un standard. Unitățile de măsură sunt purtate exclusiv de vocabularul QUDT; un termen local introdus inițial pentru simbolul unității a fost eliminat ca redundant, întrucât IRI-ul QUDT poartă deja atât simbolul, cât și o denumire lizibilă.

Al doilea: se folosește vocabularul WoT acolo unde acesta acoperă nevoia. Limitele sunt exprimate ca minim și maxim de schemă, valorile permise ca enumerare, explicațiile ca descriere de schemă — toate standard, din modulul JSON Schema in RDF (W3C, 2020b), în timp ce formularele de invocare urmează vocabularul de controale hipermedia (W3C, 2020a). Un termen local se introduce doar când nu există echivalent, cum este cazul unității, pe care specificația TD 1.1 (W3C, 2023) o definește doar ca termen de context JSON, fără un IRI utilizabil în RDF.

Al treilea: adnotările Matter rămân proveniență, nu identitate. Numele de cluster și de atribut au fost eliminate din documentele servite odată ce numele capabilității a devenit unic în interiorul unui Thing, iar SHTD a preluat rezolvarea internă a coordonatelor Matter. A supraviețuit o singură adnotare de tip, pentru că distinge unități care altfel ar fi identice în schemă — o valoare în miliwați și un întreg fără unitate arată la fel dacă tipul nu este consemnat.

Coerența acestui strat este verificată automat: fiecare termen HomeOnt emis în grafurile servite trebuie să fie definit în ontologie.

---

## 7. Procedura de configurare a evaluatorului

### 7.1 Modelul de worker

Evaluarea a șase sute de episoade impune rulare paralelă, iar aici intervine o constrângere reală a simulatorului: locuința este stare la nivel de modul, deci **un proces deservește exact o locuință**. Nu există izolare în interiorul unui proces — dar izolarea între procese este completă, ceea ce face ca modelul corect să fie evident.

Fiecare worker primește deci un simulator propriu și o proiecție TD proprie, pe porturi alocate dinamic, și rulează în serie episoadele care i-au fost atribuite, resetând simulatorul între ele. Ambele procese sunt de lungă durată, ceea ce amortizează costul pornirii pe multe episoade.

Izolarea a fost măsurată, nu presupusă. Doi workeri rulați simultan la rate de ceas diferite au raportat numere de tick-uri divergente și timpi simulați diferiți, cu configurații de locuință distincte. Proba decisivă a fost însă alta: acționarea unui bec într-un worker a dublat iluminarea camerei respective, în timp ce camera omonimă din celălalt worker a rămas complet neschimbată. Ceasurile și dinamicile de mediu sunt, așadar, independente — exact condiția cerută.

### 7.2 Curățarea abonamentelor între episoade

Abonamentele la notificări supraviețuiesc procesului agentului dacă nu sunt anulate explicit, iar ascultătorul de notificări se leagă la un port fix. Combinația celor două a produs un efect observat concret în testare: abonamente rămase dintr-o rulare anterioară continuau să livreze într-una nouă, trezind agentul pentru dispozitive la care nu se abonase niciodată.

Încheierea unui episod aplică de aceea două straturi de curățare. Întâi se cere motorului agentului să se dezaboneze corect, prin protocolul WebSub, artefact cu artefact — ceea ce exercită traseul folosit în producție și lasă registrul consistent. Apoi se repornește proiecția TD indiferent de rezultatul primului pas, întrucât abonamentele trăiesc în acel proces, iar repornirea garantează că niciunul nu supraviețuiește chiar dacă agentul a eșuat la mijlocul episodului.

Verificarea s-a făcut pe episoade consecutive rulate pe același worker, cu porturi reutilizate și cu un episod repetat: fiecare episod începe cu registrul gol, toate artefactele se abonează, iar dezabonarea readuce registrul la zero.

### 7.3 Contractul cu evaluatorul SimuHome

Evaluatorul primește un client al simulatorului și apelează pe el două metode, prin nume: una de resetare și una de avans rapid. Ambele au fost adăugate clientului nostru, returnând plicul brut al răspunsului, întrucât evaluatorul îl despachetează el însuși.

Payload-ul cerut cuprinde episodul complet, clientul, starea finală a locuinței și lista instrumentelor invocate, fiecare cu rezultatul său. Ordinea operațiilor contează, din cauza naturii contrafactuale descrise în secțiunea 2.3: starea finală trebuie captată înainte ca evaluatorul să reseteze simulatorul pentru a-și calcula linia de bază.

### 7.4 O dependență de mediu

Importul evaluatorului atrage întregul stack de agenți LLM al SimuHome. Traseul pentru episoadele fezabile de tip qt2 este însă complet determinist și nu apelează niciodată un judecător bazat pe model de limbaj — doar traseul pentru episoadele nefezabile o face, întrucât judecarea unui răspuns de tip „nu se poate" este intrinsec o sarcină de limbaj. Pentru verificările automate, furnizorul LLM este substituit cu un obiect fictiv, ceea ce permite rularea scorării deterministe fără acele dependențe. Episoadele nefezabile vor necesita instalarea lor.

### 7.5 Verificare end-to-end

Lanțul complet a fost validat pe un episod real: acțiuni trimise **prin Thing Description-uri** au satisfăcut obiectivele episodului, iar evaluatorul propriu al SimuHome a returnat scorul maxim, cu ambele camere vizate înregistrând creșterea de iluminare cerută față de linia de bază contrafactuală.

---

## 8. Stare curentă și verificare

Fiecare fază a lucrării dispune de o suită de verificare automată, rulată împotriva unei perechi simulator + proiecție TD active. Sunt acoperite astfel: bucla de bază de acționare, reprezentarea read-only a ierarhiei platformă–locuință–cameră–artefact, capabilitățile de acțiune împreună cu notificarea schimbărilor, semantica TD-SOSA cu reconcilierea față de simulator, acoperirea completă a dispecerizării comenzilor și, în fine, modelul de worker cu izolarea și scorarea sa. Toate trec integral.

La acestea se adaugă suita de teste unitare a stratului de agenți, care trece neschimbată, precum și verificările registrului Matter și ale tabelelor de mapare.

Recuperarea față de traseul Home Assistant a fost verificată pe episodul cel mai defavorabil pentru acesta din tot corpusul: toate aparatele de climatizare și toate jaluzelele apar ca artefacte, corect tipizate și cu setul complet de proprietăți. Etichetele modurilor de funcționare supraviețuiesc intacte, rezolvând limitarea de lungime care le transforma anterior în valori necunoscute.

---

## 9. Lucrări rămase

Harness-ul de rulare propriu-zis — distribuirea episoadelor pe workeri și rularea planificatorului BehaviorTree în fiecare — depinde de modificările în curs ale pipeline-ului de agenți și rămâne de finalizat.

O serie de ajustări datorate stratului de agenți au fost consemnate separat pe măsură ce au fost descoperite. Cea mai substanțială privește suprapunerea dintre identificatorul unei capabilități și URL-ul ei de invocare: numele ar trebui să provină din câmpul dedicat al Thing Description-ului, iar ținta din formularul de invocare, în loc ca ambele să fie deduse dintr-un singur șir. Se adaugă faptul că punctul de intrare al agenților ignoră setarea de mediu curent și folosește în schimb prima intrare din configurație, precum și o fabrică de motoare de integrare rămasă nefinalizată.

Episoadele nefezabile necesită, cum s-a arătat, judecători bazați pe modele de limbaj.

În fine, dacă simulatorul va fi extins cu dinamicile pe care momentan nu le modelează — efectul de răcorire al ventilatorului și lumina naturală prin jaluzele — cele două intrări eliminate din tabelul de efecte pot fi restaurate, iar reprezentarea va reflecta automat extinderea.

---

## Referințe

Vocabularele și specificațiile pe care se sprijină reprezentarea descrisă în
acest raport. Ontologia HomeOnt și extensia TD-SOSA nu figurează aici, fiind
dezvoltate în cadrul acestei lucrări.

Brick Consortium. (2024). *Brick: A uniform metadata schema for buildings*
(Version 1.3) [Ontologie]. https://brickschema.org/schema/Brick
(accesat la 26 august 2026)

Bonino, D., & Corno, F. (2008). *DogOnt: Ontology modeling for intelligent
domotic environments* [Ontologie]. Politecnico di Torino.
http://elite.polito.it/ontologies/dogont.owl (accesat la 26 august 2026)

Connectivity Standards Alliance. (2025). *Matter specification*
(Version 1.5.1) [Specificație și model de date].
https://github.com/project-chip/connectedhomeip (accesat la 26 august 2026)

ETSI. (2020a). *SAREF: Smart Applications REFerence ontology* (Version 3.2.1)
[Ontologie]. https://saref.etsi.org/core/ (accesat la 26 august 2026)

ETSI. (2020b). *SAREF4BLDG: SAREF extension for building devices*
(Version 1.1.2) [Ontologie]. https://saref.etsi.org/saref4bldg/
(accesat la 26 august 2026)

Hyperagents. (2023). *HMAS: Hypermedia Multi-Agent Systems core ontology*
[Ontologie]. https://purl.org/hmas/ (accesat la 26 august 2026)

QUDT.org. (2024). *QUDT: Quantities, Units, Dimensions and Data Types
ontologies* [Ontologie]. http://qudt.org/schema/qudt/
(accesat la 26 august 2026)

W3C. (2017b). *Semantic Sensor Network Ontology* [Recomandare W3C].
World Wide Web Consortium. http://www.w3.org/ns/ssn/
(accesat la 26 august 2026)

W3C. (2017a). *SOSA: Sensor, Observation, Sample, and Actuator ontology*
[Recomandare W3C]. World Wide Web Consortium. http://www.w3.org/ns/sosa/
(accesat la 26 august 2026)

W3C. (2020a). *Web of Things (WoT) Thing Description — Hypermedia controls
vocabulary* [Recomandare W3C]. World Wide Web Consortium.
https://www.w3.org/2019/wot/hypermedia (accesat la 26 august 2026)

W3C. (2020b). *Web of Things (WoT) Thing Description — JSON Schema in RDF
vocabulary* [Recomandare W3C]. World Wide Web Consortium.
https://www.w3.org/2019/wot/json-schema (accesat la 26 august 2026)

W3C. (2023). *Web of Things (WoT) Thing Description 1.1* [Recomandare W3C].
World Wide Web Consortium. https://www.w3.org/2019/wot/td
(accesat la 26 august 2026)

W3C. (2018). *WebSub* [Recomandare W3C]. World Wide Web Consortium.
https://www.w3.org/TR/websub/ (accesat la 26 august 2026)
