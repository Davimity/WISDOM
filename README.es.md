# WISDOM — estructura y aprendizaje superficial de proteínas

[English](README.md) | **Español**

WISDOM construye un benchmark defendible de unión proteína–DNA, convierte estructuras proteicas en
representaciones geométricas universales, proyecta la referencia de interfaz DNA sobre esas
superficies fijas y entrena los dos primeros modelos WISDOM. «Geométrico» significa que el modelo
razona sobre grafos moleculares y superficiales. El preprocesado estructural es estrictamente
independiente del problema: las etiquetas DNA viven en catálogos y sidecars separados.

El preprocesador convierte estructuras PDB o PDBx/mmCIF en un NPZ determinista y compacto por
proteína. Un NPZ es un contenedor comprimido de arrays numéricos nombrados. Se mantiene sin pickle:
no incorpora objetos Python arbitrarios serializados que podrían ejecutar código al cargarlos. Cada
archivo combina datos atómicos normalizados, un grafo espacial/covalente, una nube fija accesible al
solvente, geometría local, un grafo superficial y relaciones superficie–átomo. La sección 4.1
construye una imagen mental en lenguaje llano antes del detalle matemático.

WISDOMv1 clasifica proteínas con logits locales débilmente supervisados. WISDOMv2 conserva intacto
ese backbone y compara reglas de pooling MIL para señales pequeñas y localizadas. Ninguna implementa
todavía las etapas posteriores de química rica, comunicación bidireccional, geometría cuasi-geodésica,
dMaSIF, contraste o modelos de lenguaje.

## 0. Índice

- [1. Inicio rápido](#1-inicio-rápido)
- [2. Instalación](#2-instalación)
  - [2.1. Requisitos](#21-requisitos)
  - [2.2. Instalación de desarrollo](#22-instalación-de-desarrollo)
- [3. Benchmark DNA-binding y anotaciones](#3-benchmark-dna-binding-y-anotaciones)
  - [3.1. Task de selección y receta de dataset](#31-task-de-selección-y-receta-de-dataset)
  - [3.2. Evidencia, contactos, conflictos y splits](#32-evidencia-contactos-conflictos-y-splits)
  - [3.3. Ground truth superficial y contrato del sidecar](#33-ground-truth-superficial-y-contrato-del-sidecar)
- [4. Preprocesado estructural](#4-preprocesado-estructural)
  - [4.1. Imagen mental y recorrido completo](#41-imagen-mental-y-recorrido-completo)
  - [4.2. Preparación, ejecución e inspección del dataset](#42-preparación-ejecución-e-inspección-del-dataset)
  - [4.3. De la entrada del manifiesto a coordenadas normalizadas](#43-de-la-entrada-del-manifiesto-a-coordenadas-normalizadas)
  - [4.4. De los átomos normalizados al grafo atómico](#44-de-los-átomos-normalizados-al-grafo-atómico)
  - [4.5. De las esferas atómicas a la geometría superficial](#45-de-las-esferas-atómicas-a-la-geometría-superficial)
  - [4.6. De los puntos superficiales al NPZ final](#46-de-los-puntos-superficiales-al-npz-final)
  - [4.7. Validación, reproducibilidad y ejecución paralela](#47-validación-reproducibilidad-y-ejecución-paralela)
  - [4.8. Arquitectura del código y tests](#48-arquitectura-del-código-y-tests)
  - [4.9. Limitaciones científicas](#49-limitaciones-científicas)
- [5. Modelos entrenables de WISDOM](#5-modelos-entrenables-de-wisdom)
  - [5.1. Manifiesto de etiquetas y batching de grafos](#51-manifiesto-de-etiquetas-y-batching-de-grafos)
  - [5.2. Modelos, ecuaciones y formas tensoriales de WISDOMv1](#52-modelos-ecuaciones-y-formas-tensoriales-de-wisdomv1)
  - [5.3. Pooling y diagnósticos de localización de WISDOMv2](#53-pooling-y-diagnósticos-de-localización-de-wisdomv2)
  - [5.4. Entrenamiento, evaluación post-run y artefactos 3D](#54-entrenamiento-evaluación-post-run-y-artefactos-3d)
- [6. Bibliografía](#6-bibliografía)

## 1. Inicio rápido

El recorrido de producción tiene dos ejecuciones deliberadamente separadas. Primero,
[`experiments/dna_select.yaml`](experiments/dna_select.yaml) es una Task ordinaria y reanudable que
descubre candidatos públicos, verifica etiquetas y estructuras, asigna splits balanceados sin fuga
homóloga y emite una selección pequeña y portable. Después,
[`experiments/dna_preprocess.yaml`](experiments/dna_preprocess.yaml) consume esa selección congelada,
construye una sola vez las superficies estructurales costosas, proyecta los sidecars de evaluación
de DNA y publica el dataset inmutable de LambdaForge. El archivo `data/dna/public-sources.json`
contiene solo procedencia fijada, no ejemplos proteicos. Una selección real es una operación pública
masiva; validar y planificar no la ejecuta.

Los dos primeros comandos crean y activan `.venv`, un entorno Python aislado que impide que los
paquetes de WISDOM alteren el resto del sistema. Sustituye `/ruta/absoluta/a/LambdaForge` por el
directorio real del checkout local. Los dos `pip install -e` instalan ambos proyectos en modo
**editable**: los cambios de código tienen efecto sin reinstalar.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "/ruta/absoluta/a/LambdaForge[adaptive-hpo,parquet]"
python -m pip install -e ".[dev]"

lf validate experiments/dna_select.yaml
lf run experiments/dna_select.yaml --dry-run

# Tras la Task de selección real, descarga su artefacto pequeño como explica la Sección 3.1.
lf validate experiments/dna_preprocess.yaml
lf datasets plan experiments/dna_preprocess.yaml --verbose
lf run experiments/dna_preprocess.yaml --dry-run
lf validate experiments/wisdom_v1.yaml
lf run experiments/wisdom_v1.yaml --dry-run
lf validate experiments/wisdom_v2.yaml
```

`validate` comprueba una Task o la receta y cada Task embebida. `datasets plan` indica si cada etapa
pesada se ejecutará o reutilizará un resultado verificado por contenido. `run` ejecuta la
configuración seleccionada y, para la receta, solo publica una ubicación inmutable cuando todas las
etapas obligatorias y el índice canónico son válidos. `lf` y `lambdaforge` son comandos equivalentes.
Una publicación local tiene esta forma:

```text
runs/datasets/published/wisdom-dna/2/<content-id-prefix>/
├── base/<structure-hash>.npz
├── structures/<source-structure-hash>.cif
├── <protein>.dna.npz
├── manifest.csv
├── local-manifest.csv
├── catalog.csv
├── {train,val,test}.txt
├── identifiers.json
├── subsets/{10pct,25pct,50pct,75pct}/
│   ├── {train,val,test,proteins}.txt
│   ├── labels.csv
│   ├── identifiers.json
│   └── manifest.csv
├── annotation-report.json
├── members.jsonl
├── dataset-artifact.json
└── ...
```

`members.jsonl` es el índice streaming autoritativo: cada línea identifica una proteína, split,
tier, etiqueta DNA global, disponibilidad de ground truth local y assets base/sidecar con checksum.
`dataset-artifact.json` guarda el ID de contenido independiente de rutas y la procedencia del build
por separado. La representación se abre sin pickle así; 4.2 explica cómo descubrir la ubicación
real en Registry y 4.4–4.6 define cada array:

```python
import json

import numpy as np

with np.load("runs/datasets/published/wisdom-dna/2/<content-id-prefix>/base/<hash>.npz",
             allow_pickle=False) as protein:
    atom_positions    = protein["atom_positions"]
    atom_edges        = protein["atom_edge_index"]
    surface_positions = protein["surface_positions"]
    metadata          = json.loads(str(protein["metadata_json"].item()))
```

## 2. Instalación

### 2.1. Requisitos

- Python 3.10 o posterior;
- LambdaForge `>=0.10.0,<0.11`, normalmente instalado desde su checkout local;
- un entorno CPU con NumPy, SciPy y Gemmi;
- acceso a Internet únicamente si una entrada PDB remota no está ya en la caché raw.

WISDOM está adaptado a LambdaForge `0.10.0`. LambdaForge es la fuente de verdad para materializar
Tasks, resolver entradas/salidas, iterar y reanudar el preprocesado, definir recetas, indexar
miembros, publicar versiones inmutables atómicamente, registrar placements, eventos, recursos y
ejecuciones. WISDOM conserva la interpretación proteica, la geometría científica, la validación
exacta de NPZ/sidecars y la visualización específica.

### 2.2. Instalación de desarrollo

Un **checkout** es una copia local de un repositorio Git. Un **commit** es la revisión exacta
registrada de esa copia. Tras sustituir los dos marcadores siguientes, los comandos crean el entorno
aislado de la sección 1 y verifican que las dependencias instaladas son compatibles.

```bash
git clone <URL del repositorio WISDOM>
cd WISDOM

python -m venv .venv
source .venv/bin/activate

python -m pip install -e "/ruta/absoluta/a/LambdaForge[adaptive-hpo,parquet]"
python -m pip install -e ".[dev]"
python -m pip check
```

`PreprocessingTask` de LambdaForge ejecuta cada flujo fuente→transformación→destino. Una
`DatasetRecipe` compila esas Tasks en un grafo de dependencias, reutiliza etapas exactas y publica
solo la raíz final seleccionada. WISDOM no implementa otro planificador, grupo de procesos, caché de
etapas, Registry ni publicador. La sección 4.7 explica esta frontera.

## 3. Benchmark DNA-binding y anotaciones

### 3.1. Task de selección y receta de dataset

La pregunta científica es si una cadena proteica seleccionada une DNA. Un archivo de coordenadas no
puede responderla por sí solo: una estructura depositada describe un experimento bajo condiciones
concretas y el DNA puede faltar aunque la proteína lo una biológicamente. WISDOM separa tres
la selección de evidencia de la geometría universal y de los targets de evaluación. La separación
también es operativa: el descubrimiento público puede tardar horas, mientras que su resultado útil
es un contrato pequeño de miembros que debe poder reutilizarse de forma independiente.

[`experiments/dna_select.yaml`](experiments/dna_select.yaml) es por ello una Task ordinaria y
reanudable. Descubre positivos y negativos públicos, verifica evidencia y la estructura mínima
necesaria, aísla contradicciones, asigna clústeres externos completos a splits, balancea cada split
principal y emite un artefacto compacto `selection`. No crea superficies ni NPZ universales.

[`experiments/dna_preprocess.yaml`](experiments/dna_preprocess.yaml) consume después ese artefacto
congelado. Su etapa `geometry` crea NPZ estructurales universales sin etiqueta desde
`selection/proteins.txt`. `annotate` une el catálogo congelado con el informe geométrico exacto,
reutiliza la caché de coordenadas de geometría, crea sidecars DNA alineados y escribe
`members.jsonl`. Solo esta receta publica `wisdom-dna@2`; un build fallido no publica ninguna
`DatasetVersion`.

Así se separan cuatro conceptos de LambdaForge 0.10.0. Una **receta** dice cómo reconstruir los datos;
un **build** es una ejecución con huellas y procedencia; una **versión** es el contenido lógico
inmutable `wisdom-dna@2`; un **placement** es una copia física local o en un clúster. Copiar bytes
verificados añade un placement, no otra versión científica. **DatasetRegistry** es la autoridad de
placements gestionados. **DataCatalog** queda para aliases, datos externos, loaders u overrides
institucionales; WISDOM ya no duplica rutas gestionadas en `data/datasets.yaml`.

El flujo de producción hace explícita la frontera del artefacto:

```bash
# Primero realizar descubrimiento público y selección.
lf validate experiments/dna_select.yaml
lf run experiments/dna_select.yaml --dry-run
lf run experiments/dna_select.yaml --on citius-ctgpgpu12

# Descargar solo el output compacto del job de selección correcto.
lf artifact list JOB_ID
lf artifact fetch JOB_ID selection --output data/dna

# Después planificar y ejecutar el costoso dataset inmutable.
lf validate experiments/dna_preprocess.yaml
lf datasets plan experiments/dna_preprocess.yaml --verbose
lf run experiments/dna_preprocess.yaml --dry-run
lf run experiments/dna_preprocess.yaml --on citius-ctgpgpu12

# En otra terminal, inspeccionar todos los jobs o seguir el log durable de este build.
lf top --history 300
lf jobs logs latest --follow

# Inspeccionar la versión inmutable y su placement local seleccionado.
lf datasets show wisdom-dna@2
lf datasets stats wisdom-dna@2
lf datasets members wisdom-dna@2 --partition split=train --limit 20
lf datasets verify wisdom-dna@2
```

LambdaForge 0.10 hace de `lf run` la entrada canónica para Tasks y recetas de dataset. El bloque
`resources` superior de la receta solicita exactamente 36 CPU, 128 GiB, ninguna GPU y 24 horas; no
hace falta repetir esos recursos por CLI. `lf datasets build` permanece como alias compatible.

`lf artifact fetch` materializa el directorio nombrado como `data/dna/selection/`. Contiene
`catalog.csv`, `identifiers.json`, `proteins.txt`, listas TXT principales/de reserva, etiquetas y
vistas diluidas. Los checkpoints de candidatos y las descargas de descubrimiento son outputs de Task
separados y deliberadamente no se descargan. La receta incorpora la selección a la huella de entrada:
unos miembros distintos invalidan geometría y anotación, mientras una selección idéntica permite a
`reuse: auto` reutilizar ambas etapas verificadas. Nunca confía en un output solo porque exista un
directorio.

`train.txt`, `val.txt` y `test.txt` son miembros lógicos con balance exacto de clases;
`proteins.txt` añade reservas para evaluación local. `identifiers.json` une cada ID con etiqueta,
split, clúster al 30 %, tier y marca de miembro lógico. La etapa final copia estos contratos al
dataset publicado. Geometría declara su caché paralela `raw` como binding de artefacto para
anotación, por lo que las estructuras elegidas no se descargan dos veces.

Para disponer del mismo dataset válido en otro clúster se copia la versión inmutable, sin repetir
descubrimiento, mapeo, geometría ni anotación:

```bash
# LambdaForge elige un placement fuente verificado y lo copia al clúster destino.
lf datasets materialize wisdom-dna@2 --on OTRO_CLUSTER --strategy replicate --apply

# O se indican explícitamente origen y destino.
lf datasets replicate wisdom-dna@2 --from citius-ctgpgpu12 --to OTRO_CLUSTER --apply
```

Ambos comandos verifican la identidad de contenido y registran otro placement de la misma versión.
Para repetir solo el preprocesado pesado en otro lugar se transfiere el artefacto `selection`
completo, no una lista de IDs sin etiquetas. Para transferir un dataset terminado, la replicación es
la operación completa.

La versión 1 inválida debe eliminarse comenzando por una previsualización. `delete` verifica el
contenido registrado, rechaza consumidores activos, muestra raíz y bytes exactos y no toca nada sin
`--apply`:

```bash
# Primero revisar cuidadosamente el plan; este comando es read-only.
lf datasets delete wisdom-dna@1 --on citius-ctgpgpu12

# Borrar solo ese placement físico verificado y desregistrar el placement.
lf datasets delete wisdom-dna@1 --on citius-ctgpgpu12 --apply

# Tras borrar físicamente todos los placements, retirar el registro vacío de la versión.
lf datasets remove wisdom-dna@1
```

No se debe ejecutar primero `datasets remove`: solo olvida metadata del Registry y deja
deliberadamente los bytes en disco, dificultando su borrado gestionado seguro. Son instrucciones;
WISDOM no elimina automáticamente datasets durante una migración.

LambdaForge 0.10.0 escribe el progreso de preprocesado en stderr, que pasa a ser el log durable del
job. Una línea inicial muestra shard, número de workers y workload; después, líneas de checkpoint
del coordinador informan de los totales `records`, `ok` y `failed`. El límite por defecto de diez
segundos evita ruido y solo el coordinador escribe estos resúmenes, de modo que los workers paralelos
no compiten al formatear el progreso. En `lf top`, se selecciona el job y se pulsa `l` para abrir el
log completo. Un job antiguo no puede adquirir mensajes retroactivamente; aparecerán en nuevos
intentos construidos con la versión compatible de LambdaForge.

`dataset.version` no es un contador de caché ni el esquema NPZ. `wisdom-dna@2` promete una identidad
inmutable de miembros y bytes y sustituye a la versión 1 incorrectamente desequilibrada. Otro cambio
intencionado de contenido o contrato requerirá versión 3 o superior; LambdaForge rechaza sobrescribir
un nombre/versión existente con otro content ID.

### 3.2. Evidencia, contactos, conflictos y splits

**Fuentes públicas fijadas.** [`data/dna/public-sources.json`](data/dna/public-sources.json) es una
entrada pequeña de reproducibilidad, no una lista de ejemplos. LambdaForge fingerprinta sus bytes
porque el contrato de `PreprocessingTask` exige que las definiciones mutables de fuentes participen
en la identidad. El comando estándar no necesita dataset privado ni fichero manual de negativos.
El manifiesto fija estos contratos:

- DyProL DNA v1, registro Zenodo 19547616, publicado el 13 de abril de 2026. Sus manifiestos
  `DNA-1022_Train` y `DNA-256_Test` contienen identificador PDB-cadena, secuencia observada y máscara
  binaria de residuos de unión de igual longitud. WISDOM descarga por rangos solo estos miembros del
  ZIP de 14,5 GB y verifica SHA-256; no descarga los ensembles. Las letras de la entrada PDB se
  normalizan, pero los ID de cadena mmCIF distinguen mayúsculas y minúsculas y por ello se conservan:
  `7S01_D` y `7S01_d`, por ejemplo, designan dos cadenas depositadas diferentes. Las claves fuente
  incluyen además base de datos, release y partición, por lo que una misma proteína comunicada por
  dos fuentes no colisiona. Las filas fuente repetidas exactamente se colapsan; dos filas distintas
  con una misma clave se rechazan antes de iniciar el mapeo estructural.
- BTD-Combo en el commit de los autores `714756450e537cebbc3d9814a1fc059758fee58b`, del 1 de
  septiembre de 2024. WISDOM toma únicamente `non_DBP` de los FASTA train/test publicados. El paper
  construye esta clase desde Swiss-Prot mediante exclusión de términos funcionales y redundancia de
  secuencia. Es inferencia biológica de alta confianza, no prueba de imposibilidad de unión.
- El snapshot BioLiP2 del 29 de marzo de 2026 sirve para detectar contradicciones, no como fuente de
  etiquetas. La API REST actual de QuickGO aporta anotaciones Gene Ontology de unión a DNA y conserva
  calificadores `NOT`. `NOT DNA binding` refuerza la procedencia, pero no es obligatorio.

**Construcción positiva.** La afirmación curada DyProL aporta la etiqueta global, su train/test
oficial aporta procedencia development/test externo y la máscara binaria aporta ground truth local
por residuo. WISDOM descarga el mmCIF experimental RCSB, elige la cadena declarada, exige alineación
exacta entre máscara, secuencia fuente y residuos observados, y conserva sus índices positivos. Si
hay DNA explícito, Gemmi también verifica el contacto; la máscara revisada sigue siendo suficiente
cuando no existen coordenadas DNA utilizables. Después, la entrada del modelo contiene solo proteína:
DNA y etiquetas residuales quedan exclusivamente para anotación de evaluación.

Sea (P) el conjunto de átomos no hidrógeno de la proteína y (D) el de las cadenas DNA. Para
(p\in P) y (d\in D), (x_p,x_d\in\mathbb{R}^3) son sus centros cartesianos en ångströms. Un par
de interfaz satisface

\[
\lVert x_p-x_d\rVert_2 \leq r_c,
\qquad r_c=4.5\ \text{Å por defecto}.
\]

La norma euclídea es la distancia recta habitual. El umbral 4.5 Å recoge contactos directos y
cercanos; no afirma que todos sean enlaces químicos. Un KD-tree indexa coordenadas próximas sin
crear una matriz densa (|P|\times|D|). Un positivo necesita evidencia positiva, átomos DNA, al
menos un par y, por defecto, dos residuos proteicos distintos en contacto. DNA lejano se rechaza
como positivo no verificado.

**Construcción negativa y filtros.** Un `non_DBP` de BTD-Combo entra primero como secuencia, no como
PDB escogido por carecer de DNA. La API oficial de secuencia RCSB debe encontrar una entidad
polimérica experimental exacta. Se inspeccionan todos los matches: se rechaza si cualquiera de sus
entradas PDB contiene un polímero DNA, si el UniProt mapeado aparece en interacciones DNA de BioLiP2
o si QuickGO informa función de unión a DNA. El mapeo falla de modo conservador si no hay estructura
exacta, UniProt, cobertura suficiente, resolución aceptable o selector de cadena inequívoco. El
catálogo conserva cada comprobación booleana, etiqueta/commit BTD y `negative_confidence=high`; no
inventa una puntuación numérica.

Si quedan varias estructuras experimentales, el ranking determinista prefiere rayos X, después
microscopía electrónica, NMR y otros métodos; dentro del método, menor resolución numérica y luego
identidad PDB/entidad/cadena lexicográfica. Cada cadena cubre al menos 80 % de la secuencia y la
resolución de rayos X/cryo-EM no supera 4 Å por defecto. Ambas clases usan PDB experimentales, por lo
que el modelo no puede resolver la tarea distinguiendo geometría PDB frente a AlphaFold.

La ausencia de DNA en una estructura PDB nunca se utiliza por sí sola como evidencia de que una
proteína no une DNA.

Afirmaciones opuestas para el mismo identificador, secuencia, grupo oficial al 90 % o grupo al 30 %
que define splits producen `conflict`. Un negativo BTD homólogo de un positivo DyProL queda en
cuarentena, nunca se resuelve silenciosamente. `conflicts.csv` conserva motivo y ambas evidencias;
`exclusions.csv`, rechazos de mapeo/calidad. Un rechazo científico documentado es un resultado
normal y no aborta la curación, pero un error inesperado del programa sí, evitando publicar un
contrato científicamente incompleto.

El profiling barato registra cadenas proteicas/totales, longitudes observada y depositada, fracción
ausente, conteos, radio de giro, extensión, ejes principales, razón de aspecto, interfaz, metadata
experimental, taxonomía, SHA-256 y fuentes. Los residuos en contacto forman un grafo disperso: se
conectan si son consecutivos o si tienen átomos pesados a no más de 8 Å. Sus componentes estiman
cantidad/tamaño de regiones y el radio de giro de sus átomos mide dispersión. Son descriptores de
profiling/QC, no ground truth superficial ni filtros automáticos de dificultad. Si (c) es la posición
media de (N) átomos proteicos,

\[
R_g=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\lVert x_i-c\rVert_2^2}.
\]

El radio de giro describe compactación global; no es una feature aprendida. Estructuras muy
alargadas pasan al tier `challenge` en vez de desaparecer silenciosamente; `core` representa la
distribución ordinaria.

Secuencias idénticas y homólogos no deben cruzar splits. WISDOM usa grupos RCSB MMseqs2 al 30 % y no
ejecuta un alineador masivo propio. Los registros test publicados por DyProL/BTD permanecen en
`test`: nunca pasan a train/validation. Clusters completos del development se reparten
aproximadamente 80/20 mediante orden SHA-256 con seed. Si un cluster development solapa el test
externo, solo permanecen sus registros externos. También se separan clusters positivos en
`validation_reserve` y `test_reserve`; las reservas nunca entrenan.

Intercalar fuentes no equivale a equilibrar clases: la mayoría de secuencias negativas BTD no tiene
una cadena experimental RCSB exacta y aceptable, mientras que muchos positivos DyProL ya nombran
una. La versión 1 superó la verificación de bytes/esquema aun con 1.210 positivos y solo 62 negativos:
la verificación demostró integridad, no que el diseño científico fuese apropiado. La versión 2 elimina
el límite de negativos, examina todos los negativos BTD curados y equilibra después de filtrar
evidencia y asignar splits sin fuga homóloga. Para el split (s), sean (n_{s,0}) y (n_{s,1}) sus
negativos y positivos aceptados. WISDOM conserva por clase

\[
m_s=\min(n_{s,0},n_{s,1}).
\]

Un orden SHA-256 con seed selecciona el prefijo de la clase mayoritaria, de forma independiente al
orden en que acaben los workers y repetible entre clústeres. Cada split principal contiene así
exactamente (m_s) negativos y (m_s) positivos; si falta una clase, la publicación falla. Esto
garantiza paridad binaria, no equidad demográfica, taxonómica ni estructural, cuyas distribuciones
siguen en el informe. Geometría y anotación usan además `on_error: fail`, por lo que ningún miembro
seleccionado puede desaparecer silenciosamente y romper la paridad posterior.

`catalog.csv` y `catalog.parquet` contienen etiqueta/split, UniProt/PDB/cadena, versión/URL/checksum
y registro fuente, longitud y grupos 30/90 %, método/resolución/taxonomía, filtros negativos,
`local_gt_expected`, complejo/cadenas DNA e índices residuales cuando aplican. Los TXT train,
validation, test, reservas y `proteins.txt` solo contienen selectores `PDBID_CHAINS`.
`identifiers.json` añade etiqueta, split, clúster, tier y estado explícito de miembro/reserva. El
JSON informa candidatos/aceptados/rechazados por clase, motivos —incluido el submuestreo de la
mayoría—, fallos PDB, balance exacto, clusters, fuentes y distribuciones estructurales.

### 3.3. Ground truth superficial y contrato del sidecar

El NPZ universal describe únicamente la proteína seleccionada. No contiene `DNA`, `label`, `target`,
`split` ni campos del benchmark; el DNA no puede filtrarse a química, geometría o features. El
anotador crea un sidecar sobre los **mismos puntos y en el mismo orden**, sin regenerar superficie ni
reescribir el base. Tiene dos rutas explícitas: distancia a la envolvente DNA si hay coordenadas de
referencia y proyección de residuos DyProL si no las hay.

Si (s'_i) es el punto centrado almacenado y (o) es `coordinate_origin`, la coordenada fuente es
(s_i=s'_i+o). Para el átomo DNA (j), (x_j) es su centro y (r_j) su radio de van der Waals
tabulado por Gemmi. La separación física es

\[
d_i=\min_j\left(\lVert s_i-x_j\rVert_2-r_j\right).
\]

Restar el radio transforma distancia entre centros en una aproximación a la distancia desde el
punto superficial a la envolvente de van der Waals del DNA. Con (a=1.4) Å y (b=3.0) Å:

\[
y_i^{hard}=\mathbb{1}[d_i\leq a],\qquad
m_i=\mathbb{1}[d_i\leq a\ \lor\ d_i\geq b].
\]

(mathbb{1}) vale uno cuando se cumple la condición. `surface_target_hard` almacena
(y_i^{hard}) y `surface_valid_mask`, (m_i). Los puntos (a<d_i<b) forman una banda ambigua:
siguen visibles pero no participan en métricas binarias. Definiendo
(t_i=\operatorname{clip}((d_i-a)/(b-a),0,1)), el target continuo es

\[
y_i^{soft}=\frac{1+\cos(\pi t_i)}{2}.
\]

Vale uno en la interfaz segura, cero más allá del negativo seguro y cambia suavemente entre ambos.
El sidecar guarda además distancia, máscara de distancia válida, targets para umbrales de
sensibilidad, los umbrales, JSON de procedencia y SHA-256 del NPZ base. Los negativos curados tienen
target cero en todo punto válido. Su distancia a DNA no puede calcularse: aparece como NaN solo
donde `surface_distance_valid` es falso, nunca como una distancia cero ficticia.

Para proyectar la máscara, sea (q_i) el punto superficial (i), (A(i)) los átomos conectados a
él por el grafo disperso átomo–superficie y (ho(a)) el índice residual desde cero del átomo (a):

\[
a_i^*=\arg\min_{a\in A(i)}\lVert q_i-x_a\rVert_2,
\qquad
y_i^{hard}=\mathbb{1}[\rho(a_i^*)\in B],
\]

donde (B) son las posiciones `1` de DyProL. Así se transfiere una región curada a la discretización
fija sin convertirla en input. Si falta excepcionalmente una arista, se usa el átomo proteico más
próximo. La sensibilidad por distancia no está definida para esta ruta y la metadata declara
`local_gt_method=binding_residue_mask`.

Elegibilidad global y local son conceptos distintos. Un positivo fiable puede entrenar aunque
`local_gt_expected=false`. Tras anotar, `local_gt_available` solo es cierto si un positivo proyecta
al menos un punto positivo. Si produce cero, conserva `label=1`, invalida todos sus puntos locales,
registra `zero_positive_surface_points` y queda fuera de métricas de localización; nunca se convierte
en superficie all-negative. `manifest.csv` mantiene train/validation/test globales. Las reservas no
aparecen en él ni en `members.jsonl`; sus ficheros existen solo para sustitución local auditada.
`local-manifest.csv` contiene validation/test evaluables y sustituciones deterministas de la reserva
correspondiente; `annotation-report.json` registra original, motivo y sustituto. Los informes global
y local indican por separado sus números de proteínas.

Para curvas de aprendizaje, la Task de selección crea cada fracción indicada en `dilutions` —ahora
10 %, 25 %, 50 % y 75 %— bajo `subsets/<fracción>/`. Si el split (s) contiene (N_{sc}) ejemplos de
la clase (c), la fracción (f) conserva

\[
n_{sc}(f)=\max\left(1,\left\lfloor fN_{sc}\right\rfloor\right).
\]

independientemente para cada clase. Los splits completos ya están balanceados, así que cada vista
conserva el balance. Las filas se ordenan por amplitud entre clústeres externos al 30 %: se visita un
miembro de cada clúster antes de que alguno aporte un segundo. El orden con semilla hace las vistas
deterministas y anidadas: 10 % está contenido en 25 %, después en 50 %, 75 % y la selección completa.
Nunca se cambia el split original, por lo que permanece la barrera homóloga entre splits. La receta
pesada preprocesa la unión una sola vez; su sink de anotación convierte cada
`subsets/<fracción>/identifiers.json` en un `subsets/<fracción>/manifest.csv` ligero que referencia
los NPZ/sidecars compartidos. Para usar una vista se cambia el `subpath` de los tres loaders, por
ejemplo a `subsets/25pct/manifest.csv`; sus parámetros `split` siguen eligiendo train, validation y
test.

Con pesos de área representada (w_i>0), se guardan

\[
A_+=\sum_i w_i y_i^{hard},\qquad
A=\sum_i w_i,\qquad
f_{interface}=A_+/A.
\]

Son área positiva, área total y fracción de interfaz. Las componentes conexas positivas del grafo
superficial proporcionan `number_of_positive_regions`, permitiendo estudiar tamaño y regiones sin
cambiar el entrenamiento weakly supervised.

Una proteína globalmente positiva puede servir para entrenamiento weakly supervised aunque quede
excluida de evaluación de localización superficial porque no pueda generarse ground truth local
fiable.

El destino rechaza arrays objeto, longitudes distintas, probabilidades o máscaras inválidas, valores
finitos declarados no disponibles y huellas incompatibles antes de publicar atómicamente.
El sidecar sigue siendo pequeño. Para que el dataset sea transportable, también empaqueta una copia
idéntica byte a byte de cada NPZ universal en `annotations/base/<sha256>.npz`; nunca reescribe la
fuente. Los manifests de `annotations/` usan rutas relativas y unen `file`, `annotation`, etiqueta
global, split, identificador y tier, por lo que el dataset puede cambiar de montaje. `WisdomDataset`
vuelve a verificar la huella al cargarlo.

El `members.jsonl` final expresa la misma colección de splits principales con el modelo canónico de
LambdaForge. Cada proteína de train/validation/test es un miembro con particiones `split` y `tier`;
targets `dna_binding` y
`local_ground_truth`; estadísticas científicas de superficie; y assets `universal_npz`,
`dna_annotation` y `source_structure` con checksum. `catalog.csv`, todos los manifests, los TXT de
split, `identifiers.json` y el informe son assets globales verificados. Por ello el content ID
depende de miembros y bytes científicos, no de
la ruta en equipo o clúster. Los intermedios de selección/geometría pueden salir después de sus cachés
sin dejar incompleta la versión publicada: contiene las estructuras a las que apunta el catálogo,
los NPZ base y los sidecars.

## 4. Preprocesado estructural

El preprocesado convierte archivos de coordenadas destinados a la biología estructural en arrays
numéricos que podrá consumir un modelo de machine learning posterior. Esta sección sigue la
conversión en el mismo orden que el programa. Cada subsección presupone únicamente los conceptos
introducidos antes.

### 4.1. Imagen mental y recorrido completo

**De una proteína física a un archivo de coordenadas.**

Una proteína es una molécula física formada por átomos. Los experimentos y los programas de
predicción estructural no entregan a WISDOM la molécula física, sino un **archivo de coordenadas**.
Este archivo es una tabla estructurada que describe nombres atómicos, elementos químicos y posiciones
tridimensionales. La posición de un átomo es una terna `(x,y,z)` medida en ångströms; un ångström
(Å) equivale a `10^-10 m`.

Los archivos estructurales organizan los átomos en una jerarquía:

- un **átomo** es un elemento químico situado en una posición;
- un **residuo** es un bloque químico, normalmente un aminoácido, que contiene varios átomos;
- una **cadena** es una secuencia ordenada de residuos;
- un **modelo** es una interpretación completa de las coordenadas. Un archivo puede contener varios
  modelos, por ejemplo conformaciones experimentales alternativas.

Las dos familias de formatos admitidas son PDB y PDBx/mmCIF. Codifican ideas estructurales parecidas
con sintaxis diferentes. WISDOM delega su lectura en Gemmi en vez de interpretar manualmente el
texto. Los archivos `.gz` contienen el mismo texto PDB o mmCIF tras descomprimir gzip; gzip cambia el
tamaño de almacenamiento, no el significado molecular.

**Por qué el dataset comienza con un manifiesto TXT.**

Un dataset científico puede mezclar entradas públicas del Protein Data Bank con estructuras
privadas o generadas localmente. Copiarlas todas a un directorio único dificultaría mover, auditar y
reproducir el dataset. Por eso WISDOM parte de un pequeño **manifiesto**: un TXT donde cada línea no
comentada indica de dónde procede una proteína.

La línea puede ser un identificador PDB público como `4hhb_AB`, y entonces WISDOM puede obtener el
archivo desde RCSB PDB, o una ruta local como `../structures/model.cif.gz`. El manifiesto es la
definición ordenada del dataset; los archivos de coordenadas son sus entradas físicas. Esta separación
permite reutilizar una caché, descargar entradas públicas ausentes y notificar el fallo de una
proteína sin perder los resultados correctos de las demás.

**Por qué WISDOM crea tres grafos.**

Un modelo geométrico posterior necesita algo más que una tabla desordenada de átomos: necesita saber
qué objetos pueden intercambiar información. WISDOM representa esos intercambios posibles mediante
**grafos**. Un grafo es un conjunto de nodos y un conjunto de aristas; una arista indica que dos nodos
están relacionados. La arista no es por sí sola una fuerza química ni un mensaje aprendido, sino una
conexión estructural fija sobre la que podrá operar un modelo futuro.

WISDOM crea tres grafos complementarios:

1. El **grafo atómico** usa átomos como nodos. Une átomos próximos en el espacio, enlazados
   químicamente o ambas cosas, y describe tanto vecindarios tridimensionales como conectividad
   molecular.
2. El **grafo superficial** usa puntos muestreados de la superficie como nodos. Une puntos próximos
   que probablemente pertenecen a la misma lámina local, proporcionando un vecindario disperso sin
   construir todos los pares posibles.
3. El **grafo superficie–átomo** es bipartito: sus nodos izquierdos son puntos superficiales y los
   derechos son átomos. Los une por proximidad para que un cálculo futuro pueda transferir
   información entre el interior molecular y su frontera. «Bipartito» significa simplemente que las
   aristas cruzan entre dos tipos de nodos diferentes.

La salida NPZ contiene medidas de los nodos y estas listas de aristas, pero no activaciones de redes
neuronales, embeddings, valores de atención ni predicciones.

**El recorrido completo de los datos.**

Con esos objetos en mente, una línea del manifiesto sigue este recorrido:

```text
línea del manifiesto
    -> archivo local existente o PDBx/mmCIF descargado de forma segura
    -> modelo, cadenas, residuos y átomos seleccionados
    -> jerarquía Protein -> Chain -> Residue -> Atom
    -> arrays atómicos compactos y grafo atómico
    -> frontera accesible al solvente muestreada y geometría local
    -> grafo superficial y grafo superficie–átomo
    -> NPZ comprimido y validado más metadatos de procedencia
```

Cada flecha consume el resultado inmediatamente anterior. La sección 4.2 explica cómo ejecutar e
inspeccionar el recorrido. Las secciones 4.3–4.7 vuelven sobre las mismas flechas con detalle
científico y matemático.

### 4.2. Preparación, ejecución e inspección del dataset

El archivo asociado a la entrada lógica `protein_identifiers` contiene una estructura por línea. Un
nombre lógico es el nombre estable que usa el código; LambdaForge lo resuelve como la ruta física
declarada en `inputs`. Se ignoran líneas vacías y líneas cuyo primer carácter no blanco sea `#`. Las
líneas exactamente duplicadas se eliminan conservando la primera aparición. El orden importa porque
el informe final se restaura aunque las proteínas terminen en momentos distintos.

El preprocesador acepta un solo manifiesto `protein_identifiers` por ejecución, no una lista. En este
repositorio, `proteins.txt` es el manifiesto maestro: sus 450 identificadores únicos son exactamente
la unión disjunta de 148 identificadores de entrenamiento, 148 de validación y 154 de test. Por ello
es preferible transformar una vez el archivo maestro, no repetir el cálculo por split.

La pertenencia al split es metadato del dataset, no estructura molecular. Deben conservarse
`train.txt`, `val.txt` y `test.txt` para que la futura carga de entrenamiento seleccione los NPZ ya
procesados. No se debe deducir el nombre de salida copiando las mayúsculas del identificador: se usa
el mapa `identifier -> output` de `preprocessing-report.json`, cuyo `identifier` conserva exactamente
la línea del TXT. El manifiesto de registros de LambdaForge usa en cambio una clave estable interna
para poder reanudar y dividir registros sin confundir un nombre visible con su identidad. Los tests
verifican que los splits no se solapen y que su unión siga siendo `proteins.txt`.

**Entradas remotas.**

```text
1abc
4hhb_A
4hhb_AB
```

El código de cuatro caracteres `4hhb` es el identificador público asignado por el Protein Data Bank.
Un guion bajo introduce el selector opcional de cadenas descrito en 3.1: `4hhb_AB` significa «usar
la entrada PDB `4hhb`, pero conservar solo las cadenas `A` y `B`». Los caracteres se concatenan
porque cada uno es un ID de cadena; las comas y la antigua forma `#A,B` son inválidas. El selector de
la línea es específico de esa proteína y prevalece sobre el ajuste global `config.chains`.

**Estructuras locales.**

```text
/data/protein.pdb
/data/protein.pdb.gz
/data/protein.cif
/data/protein.mmcif
/data/protein.cif.gz
../structures/protein.mmcif.gz
```

Las rutas relativas se resuelven respecto al TXT. El nombre de un archivo local es opaco: `_AB` en
el nombre **no** selecciona cadenas. Para archivos locales se usa la configuración global `chains`.
Cada archivo local, o un directorio padre que lo contenga, debe aparecer además en `inputs` del YAML;
declararlo indica a LambdaForge de qué bytes externos depende el cálculo. Así puede incorporarlos a
la **huella de la tarea**, una identidad estable de esta entrada y configuración exactas, en vez de
reutilizar silenciosamente un resultado generado con otras coordenadas.

Solo se aceptan `.pdb`, `.cif`, `.mmcif` y sus variantes comprimidas con gzip. BinaryCIF, MMTF, XML,
trayectorias y contenedores de archivos quedan fuera del contrato actual.

**Configuración y ejecución.**

La etapa `geometry` de
[`experiments/dna_preprocess.yaml`](experiments/dna_preprocess.yaml) es la descripción estructural
editable. Esta Task embebida selecciona el manifiesto enlazado, directorios, parámetros científicos
y política de ejecución. LambdaForge la materializa como una etapa de receta direccionada por
contenido.

```bash
lf validate experiments/dna_preprocess.yaml
lf datasets plan experiments/dna_preprocess.yaml --verbose
lf run experiments/dna_preprocess.yaml --dry-run
lf run experiments/dna_preprocess.yaml
```

`validate` detecta errores de receta/etapa y clases Python no disponibles. El plan verbose muestra
`EXECUTE`, `REUSE` o `MISSING` por etapa y `PUBLISH` o `NOOP` en la frontera final. `--dry-run` no
envía jobs ni transforma proteínas. El último comando realiza el recorrido completo de 4.1 y solo
publica después de validar el índice final.

Los tres conceptos de preprocesado tienen papeles acotados. `ProteinSource` lee la entrada nombrada
`protein_identifiers` y asigna una clave estable a cada línea TXT única. `PreprocessPipeline` es la
transformación: recibe una línea y devuelve en memoria una representación validada de una proteína.
`ProteinSink` es la única frontera de publicación: escribe el NPZ atómicamente, decide si uno existente
es reutilizable científicamente y genera el informe propio de proteínas. LambdaForge rodea esas tres
clases con iteración, procesos, checkpoints, errores, manifiestos e identidad final del dataset.

**Referencia de configuración.**

| Parámetro | Default | Significado |
|---|---:|---|
| `geometry.inputs.protein_identifiers` | `data/dna/selection/proteins.txt` | Manifiesto exacto descargado de la Task de selección independiente. |
| `outputs.downloads` | `raw` | Caché nombrada y relativa al run para los `cif.gz` descargados. |
| `outputs.processed` | `processed` | Directorio nombrado del dataset NPZ. |
| `outputs.report` | `preprocessing-report.json` | Informe científico/de compatibilidad propio de WISDOM. |
| `task.params.workers` | `36` | Un worker CPU concurrente por núcleo solicitado por el comando de producción `lf run`. |
| `task.params.workload` | `cpu` | Procesos creados para transformaciones CPU; el padre escribe destino y manifiesto. |
| `task.params.on_error` | `fail` | Se detiene si un miembro balanceado elegido no produce su representación pesada. |
| `task.params.checkpoint_interval` | `1` | Persiste el progreso por registro tras cada proteína terminada. |
| `task.params.progress_interval_seconds` | `10.0` (default del framework) | Emite desde el coordinador una línea agregada de progreso en checkpoints elegibles aproximadamente cada diez segundos. |
| `transform.download` | `true` | Permite descargar de RCSB si falta una entrada remota en la caché nombrada. |
| `resources` de receta | `36 CPU, 128 GiB, 24 h` | Asignación exterior exacta compartida secuencialmente por geometría y anotación. |
| `model_index` | `0` | Modelo estructural, indexado desde cero. |
| `chains` | `[]` | Filtro global de cadenas, sustituido por un selector remoto. |
| `include_hydrogens` | `false` | Conserva hidrógenos explícitos; nunca los inventa. |
| `include_waters` | `false` | Conserva residuos de agua cristalográfica. |
| `include_nonpolymer` | `false` | Conserva residuos no poliméricos como ligandos. |
| `include_metals` | `false` | Conserva átomos/residuos metálicos. |
| `center_coordinates` | `true` | Resta el centroide filtrado y lo guarda como procedencia. |
| `atom_radius` | `6.0 Å` | Corte del grafo atómico espacial. |
| `surface_resolution` | `1.0 Å` | Escala principal de muestreo superficial, denotada `h`. |
| `probe_radius` | `1.4 Å` | Radio sumado a vdW para la superficie accesible al solvente. |
| `atom_surface_radius` | `6.0 Å` | Corte de comunicación superficie–átomo. |
| `curvature_scales` | `[2.5, 5.0]` | Multiplicadores positivos de radio para cada triplete de curvatura. |

`surface_resolution` controla densidad de candidatos, voxel y radio del grafo superficial. Las
escalas de curvatura sí son configurables: un valor `s` ajusta un triplete `[H,K,C]` dentro del radio
`s h`, donde `h=surface_resolution`. Añadir o quitar escalas cambia `surface_curvatures` de
`[M,S,3]` al nuevo número `S`; el YAML de WISDOMv1 debe fijar entonces
`curvature_features=3S` y la entrada de la proyección a `hidden_dim+3S`.

Los campos de ejecución y los científicos están separados deliberadamente. Cambiar `workers`,
`workload`, la frecuencia de checkpoint o los recursos solicitados cambia cómo se planifican los
mismos registros; no debe cambiar sus bytes NPZ ni la identidad del dataset. Cambiar un campo
científico modifica la geometría e invalida la reutilización. `PreprocessConfig` ya no contiene
rutas, números de workers, flags de reanudación ni política de fallos.

**Inspección del resultado.**

Cada entrada correcta genera exactamente un **NPZ**, un contenedor comprimido con varios arrays
NumPy nombrados en un solo archivo. La sección 4.6 describe cada array. El archivo de texto separado
`preprocessing-report.json` contiene registros
ordenados con estado (`processed`, `skipped` o `failed`), tiempo, bytes de arrays, bytes comprimidos,
tamaños de grafos y superficie y avisos. Un fallo registra identificador, tipo de excepción y
mensaje sin eliminar resultados correctos.

`processed` significa que se construyó un NPZ nuevo; `skipped`, que las comprobaciones de reanudación
de 3.7 aceptaron uno científicamente compatible; y `failed`, que solo esa línea no pudo terminar. El
JSON interno contiene la **procedencia**: un registro de auditoría que indica de dónde salieron las
coordenadas, qué ajustes las transformaron y qué versiones de software hicieron el trabajo. La
procedencia no cambia la geometría; permite rastrearla.

Los nombres hexadecimales de los directorios no son temporales arbitrarios. Cada uno es la primera
parte de una huella SHA-256 para una combinación exacta de definición de tarea y bytes de entrada
declarados. Separar las huellas en directorios de aspecto inmutable impide que un experimento nuevo
sobrescriba silenciosamente evidencia científica anterior. Estos comandos producen un índice legible
y una página HTML, evitando abrir cada directorio a mano:

```bash
lambdaforge results runs/tasks --no-archived
lambdaforge dashboard runs/tasks --output runs/index.html
```

La lista muestra intento, estado, huella y ruta de `result.json`; el dashboard agrupa visualmente la
misma información. Cuando un resultado se use para entrenar o publicar, debe conservarse su huella
exacta, no un alias móvil «latest». Los archivos `config.yaml` y `result.json` del directorio explican
qué lo produjo.

```python
import json

import numpy as np

with np.load("protein.npz", allow_pickle=False) as archive:
    print(archive.files)
    print("átomos:", archive["atom_positions"].shape[0])
    print("aristas atómicas:", archive["atom_edge_index"].shape[1])
    print("puntos superficiales:", archive["surface_positions"].shape[0])
    print("aristas superficiales:", archive["surface_edge_index"].shape[1])
    print(json.loads(str(archive["metadata_json"].item())))
```

LambdaForge 0.10.0 puede inspeccionar, restringir y dibujar arrays explícitos de forma segura sin
código de depuración propio de WISDOM. Por ejemplo:

```bash
lambdaforge artifact inspect protein.npz --json
lambdaforge artifact inspect protein.npz --array surface_curvatures --rows 20
lambdaforge artifact validate protein.npz --require-array surface_positions \
  --shape 'surface_positions=*,3' --finite
lambdaforge artifact visualize protein.npz --type point-cloud \
  --positions surface_positions --output surface.html
lambdaforge artifact visualize protein.npz --type graph \
  --nodes surface_positions --edges surface_edge_index --output surface-graph.svg
```

Estas herramientas genéricas responden qué contiene un array y si cumple restricciones explícitas
de forma y finitud. No conocen química de proteínas, distancia superficial firmada, orientación de
normales, identidades de curvatura ni la relación entre grafos atómico y superficial. El validador y
el visor independientes de WISDOM descritos en 4.7 siguen siendo necesarios para esas preguntas.

### 4.3. De la entrada del manifiesto a coordenadas normalizadas

**Lenguaje matemático común.**

Las siguientes secciones describen con detalle la transformación anticipada en 4.1. Usan varias
veces una pequeña cantidad de notación, que se introduce aquí antes de depender de ella.

Una letra minúscula en negrita, como `x` o `y`, representa un punto tridimensional. Sus componentes
`x_1`, `x_2` y `x_3` corresponden a los ejes cartesianos x, y, z. Coordenadas, distancias y radios se
miden siempre en ångströms.

La distancia en línea recta o **distancia euclídea** entre `x` e `y` es

```math
d(\mathbf{x},\mathbf{y}) = \lVert \mathbf{x}-\mathbf{y}\rVert_2
                           = \sqrt{\sum_{q=1}^{3}(x_q-y_q)^2}.
```

Aquí `q` recorre los tres ejes. En cada uno se eleva al cuadrado la diferencia, se suman los tres
cuadrados y la raíz devuelve el resultado a unidades de distancia. Las barras dobles con subíndice
`2`, `||.||_2`, son una abreviatura de esta operación.

Muchas reglas posteriores preguntan «¿qué puntos están dentro de un radio?». Recalcular la distancia
de cada par exigiría una tabla cuadrada que crece rápidamente. WISDOM usa un **KD-tree**, un índice
espacial que divide el espacio tridimensional para localizar candidatos cercanos sin enumerarlos
todos. Solo mejora la eficiencia de búsqueda: cada arista o vecindario se acepta usando la distancia
euclídea anterior.

**Obtención de los archivos de coordenadas.**

La primera etapa convierte cada línea del manifiesto de 3.2 en un archivo local legible. Una ruta
local ya cumple el requisito. Un identificador público puede exigir una descarga, por lo que WISDOM
mantiene una caché: un directorio de archivos descargados que ejecuciones posteriores pueden
reutilizar.

Los identificadores remotos se normalizan a minúsculas para la caché y se resuelven como
`downloads/<pdb_id>.cif.gz`, donde `downloads` es la ruta física de la salida nombrada. Las entradas
ausentes se descargan desde
`https://files.rcsb.org/download/<PDB_ID>.cif.gz`, endpoint PDBx/mmCIF comprimido de RCSB PDB.

LambdaForge y WISDOM dividen el trabajo así:

1. `ProteinSource` analiza y deduplica las líneas y emite claves estables;
2. LambdaForge planifica cada registro según `workers` y `workload`;
3. la transformación resuelve la ruta local o descarga la entrada remota ausente;
4. WISDOM calcula SHA-256 de la fuente y realiza la transformación científica;
5. el proceso padre de LambdaForge entrega cada registro terminado a `ProteinSink` y lo registra.

Las descargas comparten por tanto la concurrencia acotada de las transformaciones; WISDOM no mantiene
un grupo oculto de hilos de descarga. Una entrada en caché vuelve inmediatamente. Registros de varias
cadenas del mismo PDB pueden alcanzar la caché a la vez, por lo que sigue siendo necesario el
protocolo de escritor único descrito a continuación.

Dos ejecuciones podrían pedir simultáneamente el mismo PDB ausente. Un pequeño archivo `.lock` actúa
como señal de escritor único: su creación exclusiva mediante `O_EXCL` permite que solo una descargue.
Las demás comprueban cada `0.1 s` si apareció el destino terminado y abandonan tras `180 s` en vez de
esperar indefinidamente.

La descarga no escribe directamente en el nombre final. Transmite bloques HTTP de `1 MiB` a un
temporal único. `flush` y `fsync` piden al sistema operativo terminar de escribir los bytes; abrir el
resultado mediante gzip verifica que se descomprime a contenido no vacío. Solo entonces `os.replace`
renombra el temporal al destino de forma atómica. Otro proceso ve o bien ningún archivo o bien uno
completo, nunca una descarga parcial.

La etiqueta de formato se infiere del sufijo y Gemmi interpreta las coordenadas. Finalmente WISDOM
calcula SHA-256, una huella de contenido que transforma los bytes exactos en un valor hexadecimal de
longitud fija. Si cambia un byte, se espera que cambie la huella, y 3.7 puede rechazar resultados
obsoletos. En una entrada comprimida se hashean los bytes comprimidos realmente suministrados.
LambdaForge hashea entradas estáticas declaradas para la huella de la tarea; el hash por fuente de
WISDOM cubre además rutas resueltas dinámicamente y coordenadas descargadas.

**La frontera de Gemmi.**

Gemmi es una biblioteca de biología estructural que comprende la sintaxis y los diccionarios de PDB
y PDBx/mmCIF. Tras descomprimir gzip si hace falta, expone elementos, modelos, cadenas, residuos,
átomos, coordenadas, cargas y conexiones mediante una interfaz común. Esta frontera evita que los
muchos casos límite del parseo contaminen la representación científica de WISDOM.

Tras la lectura no se guarda ningún objeto Gemmi. La información seleccionada se copia a la jerarquía
simple `Protein -> Chain -> Residue -> Atom`. Los datos de auditoría se colocan en
`ProteinMetadata`; «metadatos» significa información *sobre* la representación —como hash de fuente
y origen de coordenadas—, no átomos pertenecientes a la molécula.

**Modelos, cadenas, residuos y filtrado.**

El lector reduce ahora el archivo a la molécula solicitada. Primero valida `model_index`: el valor
predeterminado `0` elige el primer modelo completo y WISDOM no promedia modelos experimentales.
Después enumera las cadenas y rechaza una solicitada que no exista; devolver silenciosamente otra
cadena cambiaría el significado del dataset.

Dentro de esas cadenas, un **residuo polimérico** forma parte de la cadena enlazada de aminoácidos o
ácidos nucleicos. Un **residuo no polimérico** es un componente separado, como un ligando. Agua,
iones metálicos, componentes no poliméricos e hidrógenos pueden conservarse o retirarse por
configuración. Lo retirado no contribuye a geometría ni grafos. WISDOM nunca inventa átomos ni repara
residuos incompletos.

Gemmi reconoce el agua y su categoría `EntityType.Polymer` identifica polímeros. Los metales se
reconocen con su tabla periódica más el conjunto de respaldo de `chemical_data.py`. Un residuo debe
tener número de secuencia: junto con cadena y código de inserción, forma la dirección usada después
para reconectar enlaces a los átomos correctos.

**Posiciones atómicas alternativas.**

Una estructura cristalina puede registrar varias posiciones observadas para un átomo del mismo
nombre. El código **altLoc** etiqueta esas alternativas. La **ocupación** estima la fracción de
moléculas representada por cada alternativa. Como los arrays posteriores necesitan una posición por
átomo, WISDOM conserva un candidato por nombre y lo ordena por:

1. mayor ocupación;
2. código altLoc blanco o `A` frente al resto;
3. blanco antes de `A` y después orden alfabético.

La primera regla elige la observación más frecuente. Blanco y `A` son posiciones principales
convencionales y ganan un empate; el orden alfabético resuelve el resto de forma repetible. Después
se ordenan los nombres dentro de cada residuo, mientras cadenas y residuos conservan el orden fuente.
Los índices resultan deterministas: `0,1,...,N-1` al recorrer la jerarquía.

**Centrado de coordenadas.**

Un archivo coloca la molécula en un marco global arbitrario: trasladar todos los átomos por el mismo
vector cambia sus coordenadas, pero no la forma ni las distancias internas. El centrado elimina esa
traslación irrelevante y mantiene magnitudes numéricas cercanas a la molécula.

Sea `N` el número de átomos conservados y `x_i` la coordenada tridimensional fuente del átomo `i`,
con `i` entre `1` y `N`. Primero se calcula el centroide `o`, promedio por componentes de todas las
posiciones. Después se resta el mismo centroide a cada átomo:

```math
\mathbf{o} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{x}_i,
\qquad
\mathbf{x}'_i = \mathbf{x}_i - \mathbf{o}.
```

La prima en `x'_i` significa «coordenada centrada», no otro átomo. Como se resta el mismo `o`, las
distancias por pares no cambian. `coordinate_origin` guarda `o` como procedencia; sumarlo de nuevo,
`x_i=x'_i+o`, recupera el marco fuente. Si se desactiva el centrado, las coordenadas quedan intactas
y `(0,0,0)` indica que no hubo traslación.

Toda coordenada debe ser finita: NaN e infinito no representan puntos físicos ni distancias válidas.
Debe quedar al menos un número atómico mayor que uno, asegurando que la estructura filtrada no sea
solo una colección aislada de hidrógenos.

**Conexiones declaradas por la fuente.**

Algunos archivos declaran que dos direcciones atómicas están conectadas. WISDOM transforma cada
`(cadena, número de residuo, código de inserción, nombre de átomo)` al índice consecutivo anterior.
Si un extremo fue filtrado, la conexión ya no puede representarse y se omite.

Los términos químicos relevantes son:

- un **enlace covalente** comparte densidad electrónica y forma el esqueleto químico molecular;
- un **enlace disulfuro** es una unión covalente S–S entre azufres de dos cisteínas;
- la **coordinación metálica** asocia un ion metálico con átomos donantes cercanos, pero aquí no
  recibe un orden covalente entero ordinario;
- un **puente de hidrógeno** es una interacción no covalente direccional entre donante y aceptor. Es
  evidencia útil de la fuente, pero no forma parte de la topología covalente de WISDOM.

`ConnectionType` y `BondType` son enums: tablas cerradas que almacenan estos significados como
categorías enteras compactas y validadas, no como texto libre propenso a errores. En mmCIF, la
columna `_struct_conn.pdbx_value_order` usa `sing`, `doub`, `trip` y `arom` para orden simple, doble,
triple y aromático. WISDOM traduce cada código a su enum. Los puentes de hidrógeno quedan en la
procedencia normalizada, pero se excluyen deliberadamente de las aristas covalentes de 3.4.

### 4.4. De los átomos normalizados al grafo atómico

**Conversión de la jerarquía en arrays atómicos.**

La jerarquía construida antes en 3.3 es natural para representar propiedad molecular, pero las bibliotecas numéricas
trabajan eficientemente con arrays rectangulares. `AtomicStructureBuilder` la recorre una vez y
escribe una fila por átomo. `N` sigue siendo el número total de átomos; forma `[N,3]` significa `N`
filas con tres coordenadas y `[N]` un valor por átomo.

Cada átomo ya tiene un índice consecutivo único. Durante el recorrido también se asignan índices
consecutivos de residuo y cadena, permitiendo rastrear cada fila sin duplicar átomos en clases padre.

Los veinte aminoácidos estándar reciben IDs `1..20` según el orden fijo de `AMINO_ACIDS`; `0` indica
desconocido, ligando u otro residuo no canónico. Son categorías, no magnitudes medidas. El rol atómico
usa esta precedencia:

1. hidrógeno;
2. metal;
3. agua;
4. no-polímero;
5. nombre de backbone (`N`, `CA`, `C`, `O`, `OXT`);
6. cadena lateral.

El **número atómico** es la cantidad de protones que identifica al elemento. La **carga formal** es
la carga entera de contabilidad escrita por la fuente. El radio van der Waals aproxima el espacio en
contacto no enlazado; el radio covalente aproxima el tamaño al juzgar separaciones enlazadas. Gemmi
aporta ambas tablas. Son entradas geométricas prácticas, no características aprendidas ni una afirmación de
que exista una frontera atómica cuántica exacta y única.

La tabla usa nombres de almacenamiento NumPy. `float32` guarda una medida real en 32 bits; `int8`,
`int16` e `int32` guardan enteros con signo de rango creciente; `uint8`, enteros no negativos entre 0
y 255; y Unicode fijo, texto breve sin objetos Python arbitrarios. Así se mantiene compacto cada
archivo sin perder los rangos necesarios.

| Array | Forma | Dtype | Significado |
|---|---:|---|---|
| `atom_positions` | `[N,3]` | `float32` | Coordenadas cartesianas en Å. |
| `atomic_numbers` | `[N]` | `uint8` | Números atómicos. |
| `residue_type_ids` | `[N]` | `uint8` | Categoría canónica; cero es desconocida. |
| `atom_role_ids` | `[N]` | `uint8` | Categoría gruesa `AtomRole`. |
| `residue_indices` | `[N]` | `int32` | Residuo propietario global consecutivo. |
| `chain_indices` | `[N]` | `int16` | Cadena retenida consecutiva. |
| `formal_charges` | `[N]` | `int8` | Carga formal de entrada. |
| `vdw_radii` | `[N]` | `float32` | Radios van der Waals de Gemmi en Å. |
| `covalent_radii` | `[N]` | `float32` | Radios covalentes de Gemmi en Å. |
| `atom_names` | `[N]` | Unicode fijo | Nombres de auditoría sin `object`. |
| `residue_names` | `[N]` | Unicode fijo | Nombres de residuos para auditoría. |

**Un grafo con dos relaciones.**

Las filas atómicas recién construidas se convierten en nodos. Sea `E_spatial` el conjunto de pares no ordenados
próximos en el espacio y `E_covalent` el conjunto conectado por enlaces químicos reconstruidos. El
conjunto persistido `E_atom` es su unión:

```math
E_{atom} = E_{spatial} \cup E_{covalent}.
```

La unión significa «pertenece a uno o a ambos conjuntos». El grafo es no dirigido: `(i,j)` y `(j,i)`
representan la misma relación, y solo se guarda la versión `i<j`. Un par covalente no se descarta
porque unas coordenadas inusuales lo sitúen fuera del corte espacial.

El mismo par puede tener ambos significados. `atom_edge_relation_mask` es una máscara de bits: `1`
significa solo espacial, `2` solo covalente y `3=1+2` ambas. Así se evita duplicar pares y se conserva
la razón de su existencia.

**Aristas espaciales.**

Sea `r_a` el `atom_radius` configurado y `x_i`, `x_j` las coordenadas de los átomos `i`, `j`. Existe
arista espacial si los índices son distintos y ordenados (`i<j`) y su distancia euclídea de 3.3 no
supera `r_a`:

```math
(i,j)\in E_{spatial}
\iff i<j \text{ y } \lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2\le r_a.
```

En lenguaje ordinario, cada átomo se conecta a todos los demás dentro de una esfera de radio `r_a`.
El KD-tree enumera los pares sin tabla `N x N`. Se calculan con precisión de trabajo `float64` y se
guardan como `float32`, reduciendo a la mitad el almacenamiento sin exigir más precisión que la
normalmente justificada por las estructuras de entrada.

**Aristas covalentes y precedencia de evidencias.**

Muchos PDB no enumeran todos los enlaces covalentes ordinarios. WISDOM combina declaraciones directas
con reglas químicas conservadoras. Un diccionario indexado por el par ordenado evita duplicados. Si
varias reglas proponen el mismo par, la **precedencia** decide qué evidencia y tipo sobreviven:

1. **Registros explícitos** son conexiones escritas por la fuente y explicadas en 3.3. Los
   registros covalentes, disulfuro y coordinación metálica reciben confianza `1.00` y pueden
   sustituir inferencias porque los aportó el depositante.
2. **Plantillas canónicas** son listas fijas de pares de nombres esperados en backbone y cadena
   lateral de los veinte aminoácidos estándar; confianza `0.98`.
3. Un **enlace peptídico** conecta el carbono carbonilo `C` de un residuo con el nitrógeno `N` del
   siguiente en la misma cadena. Solo se acepta si `d(C,N)<=1.9 Å`, evitando unir una gran ruptura
   de coordenadas por mera adyacencia de secuencia; confianza `0.99`.
4. Un posible **disulfuro** une los azufres `SG` de dos cisteínas (`CYS`) separados como máximo
   `2.3 Å`; confianza `0.95`.
5. **Fallback no canónico conservador** — solo dentro de un residuo sin plantilla. Los candidatos
   usan radios covalentes `r_i^cov`, `r_j^cov` y distancia euclídea `d_ij`. Una búsqueda amplia
   encuentra pares dentro de 2.3 veces el mayor radio covalente del residuo; después se acepta si

   ```math
   0.4\ \text{Å} \le d_{ij} \le 1.15(r_i^{cov}+r_j^{cov}).
   ```

   El límite inferior rechaza coordenadas coincidentes o choques graves; el superior admite un 15%
   sobre la suma de radios. Se registran como enlaces simples con confianza `0.55` porque la geometría
   es evidencia más débil que una declaración o plantilla conocida.

Las confianzas expresan prioridad determinista, no probabilidades experimentales calibradas. Órdenes:
simple `1`, doble `2`, triple `3`, aromático `1.5`, peptídico `1`, disulfuro `1`; cero cuando no
aplica, incluida coordinación.

Cada arista registra además si comparte residuo y cadena y la separación de índices de residuo. En
una cadena, cero significa mismo residuo, uno residuos adyacentes, etc. Entre cadenas esa distancia de
secuencia no tiene significado y se guarda el mayor `int16` como **centinela**, un valor reservado
que significa «no aplicable». Estas características aportan topología sin reconstruir propiedad a
partir de nombres.

### 4.5. De las esferas atómicas a la geometría superficial

**La superficie representada por WISDOM.**

Los átomos describen materia molecular, pero muchas interacciones suceden en la frontera expuesta al
solvente. WISDOM la aproxima con una **nube de puntos**, un conjunto finito de posiciones en lugar de
una malla triangular.

Imaginemos mover el centro de una sonda esférica de tamaño parecido al agua alrededor de la molécula
sin permitir que entre en ningún átomo. Para el átomo `i`, sea `c_i` su centro, `r_i` su radio van der
Waals y `r_p` el radio de sonda. El centro debe permanecer al menos a `r_i+r_p`, por lo que se define

```math
R_i = r_i + r_p,
```

La esfera expandida sólida de `i` contiene todo punto a distancia no mayor que `R_i`. `Omega` denota
la unión —la región perteneciente al menos a una— de todas esas esferas:

```math
\Omega = \bigcup_i \{\mathbf{x}:\lVert\mathbf{x}-\mathbf{c}_i\rVert_2\le R_i\}.
```

Las llaves describen una esfera sólida y `bigcup_i` las reúne para todos los átomos. WISDOM muestrea
la frontera de `Omega`. Es una aproximación discreta a la **superficie accesible al solvente (SAS)**,
el camino accesible al *centro* de la sonda. Difiere de la **superficie excluida al solvente (SES)**,
la frontera de contacto y reentrante tocada por la sonda. WISDOM no construye sus parches esféricos y
toroidales cóncavos. La sección 4.9 retoma esta limitación.

**Separaciones firmadas de esferas.**

Para decidir si un punto está dentro o fuera de una esfera expandida, WISDOM mide una separación
firmada. Para un punto arbitrario `x`, `||x-c_i||_2` es su distancia al centro `i` y `R_i` el radio
definido justo antes. Su diferencia es

```math
g_i(\mathbf{x}) = \lVert\mathbf{x}-\mathbf{c}_i\rVert_2 - R_i,
\qquad
g(\mathbf{x}) = \min_i g_i(\mathbf{x}).
```

Para una esfera, `g_i(x)=0` significa frontera exacta, un valor positivo es distancia exterior
restante y uno negativo indica penetración. `g(x)=min_i g_i(x)` toma la menor separación entre todos
los átomos; por tanto `g(x)<=0` exactamente cuando alguna esfera contiene `x`, igual que la unión.

Es un **campo implícito** porque su nivel cero define una frontera sin listar triángulos. Fuera de la
unión es la distancia exacta a la esfera expandida más próxima. Dentro de solapes, la separación más
negativa no siempre es la ruta más corta hasta la frontera de la unión, así que WISDOM no afirma que
sea un campo de distancia firmada (SDF) exacto en todas partes. Se evalúa solo donde hace falta, no se
guarda un grid SDF tridimensional y no se extrae una malla mediante marching cubes.

**Puntos candidatos de Fibonacci.**

La frontera ya existe matemáticamente, pero el grafo necesita un número finito de nodos. Primero se
colocan direcciones candidatas alrededor de cada esfera con un patrón Fibonacci esférico. La
construcción determinista distribuye puntos aproximadamente uniformes sin aleatoriedad, de modo que
una entrada idéntica genera una salida idéntica.

Sea `h=surface_resolution`, espaciado objetivo en ångströms, y `n_i` el número de candidatos crudos
de la esfera `i`. Su área es `4*pi*R_i^2`; dividirla por el presupuesto `0.55*h^2` estima el número:

```math
n_i = \max\left(24,
      \left\lceil\frac{4\pi R_i^2}{0.55h^2}\right\rceil\right)
```

El techo redondea hacia arriba y `max(24,...)` garantiza al menos 24 direcciones incluso en esferas
pequeñas. Para `k=0,...,n_i-1`, primero se elige la coordenada vertical `z_k`. El medio paso `k+1/2`
evita los polos exactos. Después `rho_k` es el radio horizontal necesario para quedar en la esfera
unidad:

```math
z_k = 1 - \frac{2(k+1/2)}{n_i},
\qquad
\rho_k = \sqrt{\max(0,1-z_k^2)},
```

Después `gamma=pi(3-sqrt(5))` es el ángulo áureo en radianes. Avanzar por esta fracción irracional de
vuelta evita alinear repetidamente meridianos. `phi_i` es una rotación fija por átomo, para que
átomos vecinos no comiencen con el mismo patrón:

```math
\gamma = \pi(3-\sqrt{5}),
\qquad
\phi_i = 2\pi\operatorname{frac}(0.7548776662466927i),
\qquad
\theta_k = k\gamma + \phi_i.
```

`frac(a)` es la parte fraccionaria. Finalmente `u_k` combina radio horizontal, ángulo `theta_k` y
altura `z_k`; multiplicarlo por `R_i` y sumar `c_i` lo sitúa sobre la esfera expandida:

```math
\mathbf{u}_k = (\rho_k\cos\theta_k,\rho_k\sin\theta_k,z_k),
\qquad
\mathbf{p}_{ik}=\mathbf{c}_i+R_i\mathbf{u}_k.
```

En este momento todavía hay candidatos ocultos dentro de esferas vecinas. Los dos pasos siguientes
los eliminan y reducen después la densidad del muestreo.

**Eliminación de candidatos enterrados.**

`p_ik` es el candidato `k` de la esfera del átomo `i`, por lo que `i` es su **propietario**. Solo
pertenece a la frontera de la unión si ninguna otra esfera lo cubre desde el solvente. Para cada
átomo distinto `j` se comprueba

```math
\exists j\ne i:
\lVert\mathbf{p}_{ik}-\mathbf{c}_j\rVert_2 < R_j-\tau,
\qquad
\tau=\max(10^{-5},0.02h).
```

El lado izquierdo es la distancia del candidato al centro `j`; el derecho, su radio reducido por la
tolerancia `tau`. Solo se elimina si está claramente dentro, no por redondeo en dos fronteras casi
coincidentes. `tau` es el mayor entre `10^-5 Å` y el 2% de `h`. Con la separación definida antes, la misma
regla es `g_j(p_ik)<-tau`.

El KD-tree solo acelera: devuelve centros hasta el mayor radio expandido más `0.05h`; uno más lejano
no puede contener al candidato. La distancia exacta decide. Los supervivientes se llaman
**candidatos expuestos** porque el centro de la sonda puede alcanzarlos. Si no queda ninguno, no hay
superficie válida para las siguientes etapas y el proceso falla en vez de publicar una vacía.

**Reducción voxel determinista.**

Los candidatos de distintos átomos pueden agruparse en los solapes, inflando densidad y grafo. Por
eso WISDOM coloca una rejilla imaginaria y conserva como máximo uno por celda. Una celda
tridimensional también se llama **voxel**, análogo 3D de un píxel.

Sea `o` el origen formado por las menores coordenadas x, y, z expuestas. No es el centroide de 3.3;
solo ancla la rejilla. Para punto `p`, restar `o`, dividir por lado `h` y aplicar suelo da la celda
entera `q(p)`:

```math
\mathbf{q}(\mathbf{p}) =
\left\lfloor\frac{\mathbf{p}-\mathbf{o}}{h}\right\rfloor.
```

El suelo redondea cada componente hacia abajo: todos los puntos del mismo cubo `h x h x h` reciben
la misma terna. Las celdas se ordenan por x, luego y, luego z. En cada una se elige el candidato
original más cercano a `o+(q+1/2)h`; el orden original resuelve empates. No se mueve, por lo que sigue
sobre una esfera expandida.

El resultado es reproducible y contiene como máximo un punto por voxel, pero puntos de voxels
adyacentes pueden quedar cerca de su cara común. Es control de densidad, no una garantía de distancia
mínima `h` ni un muestreo Poisson-disk/blue-noise óptimo.

**Normales superficiales exteriores.**

Una **normal superficial** es un vector unitario perpendicular a la superficie local. Su orientación
debe apuntar desde la molécula hacia el solvente. En una sola esfera es la dirección normalizada del
centro al punto. En intersecciones, elegir bruscamente un propietario haría saltar las normales, por
lo que WISDOM mezcla direcciones cercanas.

Para punto `p`, se calculan las separaciones `g_j(p)` definidas antes en esta sección. `g_min` es la menor. `sigma` es una
longitud de suavizado: 25% de `h`, pero al menos `10^-3 Å` para evitar dividir por casi cero:

```math
\sigma=\max(0.25h,10^{-3}),
\qquad
g_{min}=\min_j g_j(\mathbf{p}).
```

Solo son activos los átomos con `g_j<=g_min+2.5*sigma`; esferas mucho más alejadas no influyen.
Para el átomo activo `j`, `w_j` decrece exponencialmente al separarse de `g_min` y `nabla g_j` es la
dirección radial exterior unitaria:

```math
w_j=\exp\left(-\frac{g_j-g_{min}}{\sigma}\right),
\qquad
\nabla g_j(\mathbf{p})=
\frac{\mathbf{p}-\mathbf{c}_j}{\lVert\mathbf{p}-\mathbf{c}_j\rVert_2}.
```

Se suman las direcciones ponderadas y se divide por la longitud euclídea de la suma:

```math
\mathbf{n}(\mathbf{p})=
\frac{\sum_j w_j\nabla g_j(\mathbf{p})}
     {\left\lVert\sum_j w_j\nabla g_j(\mathbf{p})\right\rVert_2}.
```

Este suavizado varía más gradualmente en intersecciones. Si contribuciones opuestas casi se cancelan,
normalizar sería indefinido y se usa la radial del propietario como alternativa determinista. Un KD-tree
considera centros hasta el mayor radio más `h`; como al eliminar candidatos enterrados, limita trabajo sin alterar la regla.

`estimate_normals` es una utilidad separada para nubes sintéticas sin esferas propietarias. Usa
vecinos en `3h` o hasta ocho próximos, resta su media y forma la matriz de dispersión `X^T X`. El
análisis de componentes principales (PCA) identifica direcciones de variación; el autovector de
menor autovalor es donde menos varía el vecindario y aproxima la perpendicular. Como el signo de un
autovector es arbitrario, una referencia exterior o regla determinista lo orienta. El `build`
molecular usa la mezcla de separaciones anterior, no estas normales PCA.

**Curvatura a escalas configurables.**

La normal indica hacia dónde mira la superficie; la **curvatura** describe cómo se dobla. Un plano
tiene curvatura cero, una esfera pequeña se curva más que una grande y una silla se dobla con signos
opuestos en dos direcciones. Como una nube muestreada no tiene un único tamaño de vecindario perfecto,
WISDOM estima radios configurables. Si `q_j` es la entrada `j` de `curvature_scales`, el radio
físico del vecindario es

```math
r_j=q_jh.
```

`h` es la resolución superficial introducida antes. El default `q=(2.5,5.0)` usa por tanto radios
`2.5h` y `5h`: el primero captura detalle local y el segundo promedia una zona mayor. El YAML puede
añadir, quitar o reordenar multiplicadores positivos únicos. Con menos de siete puntos se usan hasta
doce vecinos, evitando que el ajuste de seis parámetros quede indeterminado.

En punto `p` con normal `n`, se construyen vectores unitarios `t_1`, `t_2` perpendiculares entre sí y
a `n`. Definen el **plano tangente**, aproximación plana local. Para vecino `x`, sea
`delta=x-p`. Los productos escalares lo proyectan sobre ambas tangentes y la normal:

```math
u'=\frac{\delta\cdot\mathbf{t}_1}{r_j},
\qquad
v'=\frac{\delta\cdot\mathbf{t}_2}{r_j},
\qquad
z'=\frac{\delta\cdot\mathbf{n}}{r_j}.
```

Así `(u',v')` sitúa al vecino sobre el plano y `z'` mide altura respecto a él, todo sin unidades.
Dividir por el radio mantiene las columnas numéricamente comparables en cada escala. Se aproxima la
altura adimensional mediante un parche de Monge cuadrático:

```math
z'(u',v') \approx
\frac{1}{2}a{u'}^2+bu'v'+\frac{1}{2}c{v'}^2+du'+ev'+f
```

`a`, `b`, `c` describen flexión de segundo orden; `d`, `e` permiten inclinación residual y `f`
desplazamiento vertical. Los seis coeficientes minimizan el error cuadrático ponderado. Un vecino con
desplazamiento `delta` recibe peso gaussiano

```math
w(\delta)=\exp\left(-\frac{\lVert\delta\rVert_2^2}{r_j^2}\right).
```

por lo que los cercanos influyen más y uno a distancia `s` pesa `exp(-1)`. En notación matricial, `A`
contiene los seis términos, `beta=(a,b,c,d,e,f)` los coeficientes, `z` las alturas y la diagonal `W`
los pesos. Resolver `sqrt(W) A beta = sqrt(W) z` transforma mínimos cuadrados ponderados en
ordinarios. El solver descarta direcciones singulares menores que `0.05` veces la mayor. Este corte
del cinco por ciento impide que muestras escasas o casi colineales amplifiquen ruido hasta producir
picos de curvatura.

Para un parche pequeño casi tangente, los coeficientes de segundo orden aproximan la forma local.
WISDOM usa la matriz 2 por 2 orientada al exterior

```math
S \approx -\frac{1}{r_j}\begin{bmatrix}a&b\\b&c\end{bmatrix}.
```

El signo hace positiva una esfera convexa con normal exterior. Sus autovalores `k_1`, `k_2` son las
**curvaturas principales**, flexiones máxima y mínima en direcciones tangentes perpendiculares. Se
derivan tres canales:

```math
H=\frac{k_1+k_2}{2},
\qquad
K=k_1k_2,
\qquad
C=\sqrt{\frac{k_1^2+k_2^2}{2}}.
```

`H` es curvatura media y conserva orientación global; `K` es gaussiana, positiva si ambas direcciones
se doblan con igual signo, negativa en una silla y cero si alguna es plana; `C` es curvedness, magnitud
no negativa grande cuando alguna principal lo es.

`H` y `C` tienen unidades Å⁻¹ y el producto `K` Å⁻². Si `M` es el número de puntos y `S` el de
escalas configuradas, `surface_curvatures` tiene forma
`[M,S,3]`: punto, escala y canal `(H,K,C)`. Con defaults, `S=2` en radios `(2.5h,5h)`. Un ajuste
no finito se sustituye por cero para no persistir corrupción, pero no se convierte por ello en medida
exacta. Deben considerarse las limitaciones de 3.9.

**Pesos de área.**

La selección voxel puede dejar variaciones de densidad. Para compensarlas, se consulta cada punto y
hasta seis vecinos. Sea `ell_i` la mediana de distancias no nulas desde `i`. Elevar longitud al
cuadrado produce unidades de área. Sea `epsilon_float32` un valor positivo de seguridad que evita
peso cero si coinciden puntos. Se definen proxy crudo `A_i^*` y peso normalizado `w_i`:

```math
A_i^*=\max(\ell_i^2,\epsilon_{float32}),
\qquad
w_i=\frac{A_i^*}{\sum_j A_j^*}.
```

El denominador suma los proxies de los `M` puntos, de modo que los pesos son finitos, positivos,
adimensionales y suman uno. Una zona dispersa tiene mayor espaciado y peso. Un pooling futuro puede
evitar así que una región densa domine solo por tener más muestras. No son áreas Voronoi exactas ni
superficie accesible física (SASA) en Å²; la normalización elimina el área absoluta. Un punto único
recibe peso uno.

### 4.6. De los puntos superficiales al NPZ final

**El grafo superficial.**

Los puntos y su geometría ya existen, pero aún faltan las aristas del grafo presentado en 3.1. Conectar todos
los pares sería denso y uniría lados no relacionados. El KD-tree propone pares con distancia
`d_ij<=2.5h`, donde `h` es resolución. Después se aplican dos filtros de orientación.

Sean `n_i`, `n_j` las normales exteriores. Su producto escalar vale `1` si apuntan igual, `0` si son
perpendiculares y `-1` si son opuestas. Se rechaza cuando

```math
\mathbf{n}_i\cdot\mathbf{n}_j < -0.25,
```

para no unir láminas enfrentadas a través de un hueco estrecho. Para el segundo filtro,
`Delta_ij=p_j-p_i` es el desplazamiento de `i` a `j`. Sus productos con las normales miden cuánto
atraviesa la superficie en vez de discurrir sobre ella. Se rechaza si

```math
\max(|\Delta_{ij}\cdot\mathbf{n}_i|,
     |\Delta_{ij}\cdot\mathbf{n}_j|)
>0.8\lVert\Delta_{ij}\rVert_2,
\qquad
\Delta_{ij}=\mathbf{p}_j-\mathbf{p}_i.
```

El lado derecho es el 80% de la longitud total. La arista sobrevive solo si la componente normal no
domina desde ningún extremo. Ambas reglas favorecen recorrido tangente y reducen atajos. Los pares no
dirigidos se guardan una vez con `src<dst`, como en el grafo atómico.

Una **componente conexa** es un conjunto maximal de nodos alcanzables siguiendo aristas. Se calcula
en una adyacencia dispersa simétrica: la simetría representa ambos sentidos de cada arista y COO es
una lista eficiente de coordenadas fila-columna no nulas.

Sea `M` el número de puntos. Se avisa si las componentes superan el mayor entre 3 y `floor(M/100)`, o
si las de menos de cinco puntos superan el mayor entre 2 y la mitad del total. Esto detecta
fragmentación inusual; los IDs no afirman clasificar exterior, pockets, canales o cavidades cerradas.

«Componente» describe el **grafo**, no si los átomos correspondientes forman una única proteína. Dos
puntos de componentes distintas realmente no tienen ningún camino de aristas superficiales aceptadas
entre ellos. Puede ser físicamente correcto: dos cadenas seleccionadas quizá no se toquen y la pared
de una cavidad interna sellada está separada de la frontera exterior. También puede ser un artefacto
de discretización: el voxelizado puede dejar un pequeño hueco o los filtros de normales pueden
rechazar la única arista candidata cerca de un pliegue agudo. Una componente de un solo punto es un
punto aislado y no recibe mensajes de vecinos en el GCN superficial.

Por tanto se permiten varias componentes, pero no se declaran inocuas. El validador numérico prueba
por separado que cada punto está sobre la envolvente molecular y posee normal exterior; después
informa número de componentes, puntos aislados y fracción de la componente mayor como avisos
científicos. En el corpus actual de 450 proteínas, 4.280 de 5.389.038 puntos están aislados (`0,079%`)
y la peor fracción de componente mayor es aproximadamente `0,572`. Es geométricamente válido, pero
debe inspeccionarse antes de extraer conclusiones de localización. WISDOMv1/v2 no propagan mensajes
del GCN superficial entre componentes. Aun así pueden clasificar la proteína completa porque cada
punto conserva información de átomos próximos y el pooling MIL final reúne todas las componentes.
Si una tarea exige comunicación a larga distancia entre láminas separadas, cambiar el modelo o el
criterio de aristas es una decisión científica nueva que la validación no debe inventar en silencio.

**El grafo superficie–átomo.**

El grafo superficial mueve información por la frontera, pero un modelo futuro necesita el tercer
grafo presentado en 3.1 para comunicarse con átomos. Sea `p_s` el punto `s`, `x_i` el átomo `i` y `r_sa` el
`atom_surface_radius` configurado. Existe arista si

```math
(s,i)\in E_{surface-atom}
\iff \lVert\mathbf{p}_s-\mathbf{x}_i\rVert_2\le r_{sa},
```

Es decir, cada punto conecta con **todos** los átomos dentro de `r_sa`: es un grafo por radio, no por
número fijo de vecinos. Estos se ordenan de forma determinista. Cada columna guarda
`[surface_index,atom_index]` y su distancia euclídea.

No hay alternativa de K vecinos más próximos (KNN) que conecte un átomo lejano para completar una cuota.
Si un punto no tiene átomo dentro del corte científico, la representación es incoherente y falla. Las
aristas solo permiten comunicación futura; no contienen mensajes, atención ni pesos aprendidos.

**Esquema de salida NPZ.**

Las tres representaciones convergen ahora en un NPZ. Los arrays se separan por función para que un
consumidor cargue solo lo necesario. En la tabla, `N` es número de átomos, `M` puntos superficiales y
cada columna de aristas es un par. Un **dtype** es el tipo de almacenamiento: `float32` representa un
real de 32 bits e `int32` un entero con signo de 32 bits.

| Grupo | Arrays | Semántica |
|---|---|---|
| Átomos | `atom_positions`, `atomic_numbers`, `residue_type_ids`, `atom_role_ids`, `residue_indices`, `chain_indices`, `formal_charges`, `vdw_radii`, `covalent_radii` | Features estructurales compactas. |
| Auditoría | `atom_names`, `residue_names` | Etiquetas Unicode de ancho fijo. |
| Topología atómica | `atom_edge_index`, `atom_edge_distance`, `atom_edge_relation_mask` | Unión de pares espaciales/covalentes. |
| Enlaces | `atom_edge_bond_type`, `atom_edge_bond_order`, `atom_edge_bond_source`, `atom_edge_bond_confidence` | Tipo, orden, evidencia y confianza heurística. |
| Contexto | `atom_edge_same_residue`, `atom_edge_same_chain`, `atom_edge_residue_separation` | Contexto de propiedad/topología. |
| Superficie | `surface_positions`, `surface_normals`, `surface_curvatures`, `surface_area_weights`, `surface_component_ids` | Nube fija y geometría local. |
| Topología superficial | `surface_edge_index`, `surface_edge_distance` | Grafo local no dirigido filtrado. |
| Comunicación atómica | `surface_atom_edge_index`, `surface_atom_distance` | Incidencia bipartita por radio. |
| Provenance | `metadata_json` | Array Unicode escalar, nunca pickle/object. |

Los índices de grafos son `int32`; categorías y flags usan enteros compactos; distancias y geometría
persistida son `float32`. Se excluyen adyacencias densas, one-hot, RBF, vectores relativos,
embeddings, mensajes, parches y etiquetas específicas de modelo.

Estas exclusiones mantienen el preprocesado independiente del modelo. Una adyacencia densa es una
tabla nodo–nodo completa y casi vacía; one-hot expande una categoría en muchas columnas cero/uno; una
RBF (función de base radial) convierte una distancia en varios canales suaves; y embeddings/mensajes
son estados aprendidos por una red. El entrenamiento puede derivarlos de la estructura fija, pero
guardarlos aquí ligaría el dataset a un diseño concreto.

`metadata_json` es la procedencia introducida en 3.2: fuente/ruta/hash/formato, cadenas y modelo,
origen, counts, versiones de esquema/proyecto, configuración científica y hash, versiones de
dependencias y avisos. LambdaForge registra una sola vez por run la identidad del código y el entorno
completo, por lo que WISDOM no repite consultas al commit Git dentro de cada NPZ. Separarla de
`Protein -> Chain -> Residue -> Atom` impide confundir auditoría con estructura molecular.

### 4.7. Validación, reproducibilidad y ejecución paralela

**Validación y publicación atómica.**

Generar arrays no basta: cada etapa presupone formas e índices de la anterior. `StorageManager`
comprueba la representación completa antes de publicar el nombre final, convirtiendo corrupción
silenciosa en un fallo por proteína con motivo notificable.

Antes de publicar, `StorageManager` comprueba:

- coordenadas atómicas `[N,3]` finitas y no vacías y longitudes de características;
- números atómicos e índices de residuo válidos;
- índices `int32`, extremos en rango, `src<dst`, ausencia de duplicados y distancias coherentes;
- máscaras en `{1,2,3}` y una feature por arista;
- posiciones superficiales `[M,3]` finitas, no vacías, y normales unitarias;
- curvaturas `[M,S,3]` finitas, donde `S` coincide con el número configurado de escalas;
- pesos positivos y finitos que suman uno;
- IDs de componente no negativos;
- grafo superficial válido;
- grafo bipartito válido y al menos un átomo por punto;
- ausencia de `dtype=object`.

La publicación es **transaccional**: la ruta final solo cambia cuando el archivo nuevo entero es
válido. Se escribe un temporal único, se sincroniza, se reabre con `allow_pickle=False` y se revalidan
los bytes almacenados. Desactivar pickle impide ejecutar objetos Python serializados al cargar el
NPZ. `os.replace` lo publica atómicamente; un proceso fallido no deja un NPZ aparentemente válido.

**Reutilización de etapas y receta completa.** Una segunda invocación normal de la misma receta es
`NOOP`, no otro preprocesado. En cada etapa LambdaForge exige la misma huella de Task, resultado
correcto, rutas obligatorias y digests actuales iguales a los registrados. WISDOM revalida además
los registros moleculares reanudados en las fronteras de sus sinks. Por ello

```bash
lf datasets plan experiments/dna_preprocess.yaml --verbose
```

muestra `REUSE` en todas las etapas y `NOOP` al publicar cuando nada relevante cambia. LambdaForge
no descarga ni reconstruye superficies. Cambiar bytes, identidad del código o un ajuste científico
invalida esa etapa y sus descendientes. Cambiar contenido científico publicado exige además una
versión explícita nueva, pues un `name@version` existente es inmutable.

**Reanudación por proteína.** LambdaForge registra clave estable, estado, intentos, duración y error
de cada registro en `preprocessing-manifest.json`. Tras una interrupción ofrece las claves terminadas
al destino en vez de confiar ciegamente en sus nombres. `ProteinSink.is_complete` abre el candidato
exacto con `allow_pickle=False`, exige todos los arrays, repite las comprobaciones numéricas, vuelve
a calcular el hash del archivo de coordenadas actual y además requiere igualdad de:

```text
source_hash
config_hash
preprocessing_schema_version
```

`source_hash` identifica los bytes exactos; `config_hash`, los ajustes que cambian arrays científicos;
y `preprocessing_schema_version`, cómo se nombran e interpretan. El hash de configuración incluye
modelo/cadenas/filtros/centrado y grafos/superficie, pero no rutas, workers, descarga ni política de
fallos, porque no alteran valores. Solo se reanuda cuando coinciden ambas capas: LambdaForge tiene
progreso compatible y el destino WISDOM demuestra que el NPZ actual es reutilizable científicamente.
`--force` crea otro intento tras un éxito, `--no-resume` ignora progreso parcial y `--restart`
comienza el intento desde cero. La recuperación normal no necesita flags. El informe final conserva
el orden del manifiesto.

**Validación científica al publicar.** El digest de LambdaForge demuestra que los bytes actuales son
los registrados por la tarea, mientras el destino de preprocesado WISDOM comprueba su significado de
dominio antes de publicar cada NPZ. Lo abre sin pickle, valida el esquema completo, recalcula conteos
y distancias, verifica hashes de metadatos/configuración/fuente y contrasta el informe con los arrays.
Los avisos de fragmentación siguen visibles sin convertirse en errores de esquema. La sección 4.2
muestra los comandos genéricos de LambdaForge para inspeccionar arrays y roles 3D; no hace falta un
segundo YAML de preprocesado.

La validación superficial va más allá de comprobar arrays. Para un punto `p`, centro atómico `c_i`,
radio de van der Waals `r_i` y radio de la sonda `r_probe`, recalcula la distancia firmada respecto de
las esferas expandidas:

```math
g(p)=\min_i\left(\lVert p-c_i\rVert-r_i-r_{probe}\right).
```

La norma es la distancia euclídea ordinaria en ångströms. Una muestra de la frontera expuesta debe
tener `g(p)` próximo a cero: un valor bastante negativo sitúa el punto dentro de un átomo expandido;
uno positivo por encima de tolerancia lo deja separado o «volando». La tolerancia aceptada es
`max(0.0005 Å, 0.025h)`. El validador reconstruye además la dirección exterior de la envolvente
soft-min y exige que su coseno con la normal guardada sea al menos `0.99`. Comprueba la identidad de
curvatura `C²=2H²-K`, acota la magnitud adimensional `C r` para cada radio ajustado y registra puntos
aislados, componentes conexas, fracción de la componente mayor y arista superficial más larga. El
informe muestra estas cantidades por proteína y sus extremos en todo el dataset.

Los tests numéricos encuentran errores invisibles en una tabla; el visor 3D de artefactos con roles
explícitos de LambdaForge puede mostrar patrones espaciales ocultos por un resumen. Permite renderizar
posiciones atómicas, puntos superficiales, aristas y normales directamente desde el NPZ, como muestra
3.2. El visor no inventa una malla triangular: el preprocesado guarda una nube de puntos y unir
vecinos ingenuamente podría dibujar láminas falsas sobre cavidades.

**Paralelismo, fallos y ejecución gestionada.**

Las proteínas son registros independientes y LambdaForge puede transformar varias a la vez.
`workers: 1` es la referencia secuencial. Con `workload: cpu`, `workers` acota un grupo de procesos
creados que solo ejecutan la transformación WISDOM; el padre serializa las escrituras del destino,
checkpoints, manifiesto y publicación. `workload: io` usa hilos acotados, `workload: auto` elige hilos
conservadores y `workload: gpu` permite un worker para que el paralelismo GPU sea una decisión
explícita de jobs y recursos. Estos campos de ejecución no entran en la huella del dataset: los modos
secuencial y paralelo deben producir el mismo contenido científico.

La Task de selección solicita 36 CPU pero emplea 72 hilos de E/S acotados: dos candidatos por CPU
pueden esperar respuestas de red independientes sin sobresuscribir cálculo numérico. Un único
limitador seguro entre hilos restringe a cuatro los inicios de petición por segundo entre todos. RCSB
recomienda comenzar con solo unas pocas peticiones API por segundo y retroceder ante HTTP 429, por lo
que añadir más hilos no elevaría la tasa segura. Inspeccionar decenas de miles de candidatos todavía
puede requerir horas porque la latencia y límites del servicio remoto —no la CPU— fijan el mínimo.

La receta pesada utiliza en cambio 36 procesos creados, uno por CPU solicitada, tanto para geometría
como para anotación. Geometría descarga en paralelo diferentes entradas PDB elegidas y realiza las
costosas superficies; anotación consume el artefacto de coordenadas ya descargado. Por tanto, no
repite las descargas públicas de estructuras.

En LambdaForge 0.10.0, el bloque `resources` de la receta determina la reserva exterior real:

```bash
lf run experiments/dna_preprocess.yaml --on citius-ctgpgpu12
```

El dry-run verificado informa `CPU=36`, 128 GiB, ninguna GPU y 24 horas. `processes: 1` en ese plan no
es el número de workers de proteínas ni limita el cálculo a un núcleo: significa un coordinador
exterior del build, no varios ranks distribuidos. Dentro de ese coordinador, cada
`PreprocessingTask` crea su pool configurado de 36 procesos. Las dos etapas de la receta son
secuenciales y reutilizan la misma reserva de 36 núcleos, no requieren 72. No conviene usar 72
procesos CPU-bound con 36 CPU; la sobresuscripción suele aumentar cambios de contexto y memoria, no
rendimiento. Los 72 workers de selección son hilos y solo son apropiados porque esa Task separada
pasa la mayor parte del tiempo esperando E/S.

`on_error: skip` conservaría la clave fallida y seguiría; `on_error: fail` registra el fallo, cancela
lo pendiente y detiene la tarea. Selección, geometría y anotación eligen deliberadamente `fail`. Los
rechazos científicos son filas normales de selección y no rompen la Task, pero un error inesperado
o no producir geometría/anotación para un miembro ya equilibrado impide publicar. Los registros
terminados quedan en checkpoints para reanudar.

NumPy y SciPy pueden abrir hilos matemáticos propios. Si cada proceso crease otro grupo completo,
habría más hilos activos que CPUs asignadas: **sobresuscripción**. WISDOM fija `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` y `NUMEXPR_NUM_THREADS` a uno antes de importar cálculo en
los procesos.

En YAML de Task, Experiment y recetas de dataset LambdaForge 0.10, el bloque superior `resources`
solicita CPU, RAM, GPU, ranks de proceso y tiempo de forma portable. Tras registrar un perfil como
`atlas`, el runner gestionado prepara el build y expone resultados sin un script SLURM propio:

```bash
lf run experiments/dna_preprocess.yaml --on atlas
lf jobs show latest
lf datasets show wisdom-dna@2
```

Con LambdaForge 0.10.0, cada perfil de clúster con `environment: managed` normalmente debe dejar
automática la selección de PyTorch:

```yaml
pytorch:
  channel: auto
  require_cuda: auto
```

Antes de crear o reutilizar el entorno remoto, LambdaForge inspecciona el Python configurado en el
clúster, la arquitectura de la máquina, el driver de NVIDIA y la capacidad de cómputo de las GPU
visibles. A partir de esos datos selecciona un wheel oficial de PyTorch para CPU o CUDA que sea
compatible y, cuando se requiere GPU, verifica CUDA con una operación tensorial real. El wheel
elegido forma parte de la identidad del entorno; un cambio de compatibilidad crea un entorno
gestionado nuevo en vez de reutilizar silenciosamente uno inadecuado. Este mecanismo instala wheels
de PyTorch en el espacio del usuario: nunca modifica el driver de NVIDIA ni instala el toolkit CUDA
del sistema. Si el nodo de login de un scheduler no expone la GPU del nodo de cómputo, la detección
automática no puede demostrar compatibilidad y hace falta una política explícita revisada por el
administrador.

Las raíces físicas no tienen que coincidir entre máquinas. DatasetRegistry registra un placement
distinto para cada copia verificada de `wisdom-dna@2`, mientras el content ID no cambia. El
entrenamiento usa la referencia versionada y LambdaForge selecciona un placement en el entorno de
ejecución. DataCatalog no hace falta para esta versión gestionada; queda para aliases, datos
externos, loaders y pins institucionales explícitos.

### 4.8. Arquitectura del código y tests

**Arquitectura del código.**

El código refleja el recorrido y evita una función monolítica. LambdaForge posee la envoltura
genérica de la tarea; `PreprocessPipeline` se lee como la transformación de una proteína descrita en
3.1 y cada clase periférica encapsula un tipo cohesivo de complejidad:

`src/preprocess` usa una clase por archivo de clase y ninguna función libre. Las tablas químicas
estáticas permanecen en `chemical_data.py` sin envolverlas en estado artificial.

```text
LambdaForge PreprocessingTask
├── ProteinSource                registros TXT y claves estables
├── PreprocessPipeline           transformación científica de una proteína
│   ├── StructureCache          rutas, locks de descarga y hashing
│   ├── ProteinReader           normalización Gemmi y conexiones explícitas
│   ├── AtomicStructureBuilder  características atómicas y grafo unión
│   └── SurfaceBuilder          superficie, normales, curvatura y grafos
└── ProteinSink                   reanudación científica y publicación atómica
    └── StorageManager              esquema NPZ y validación numérica exacta

Protein
└── Chain
    └── Residue
        └── Atom
```

La procedencia definida en 3.2 se transporta en `ProteinMetadata`; ruta, hash, formato y cadenas
solicitadas en `StructureSource`. Las categorías cerradas usan enums (`AtomRole`, `BondType`,
`BondSource`, `ConnectionType`, `Relation`), que restringen valores a un conjunto documentado en vez
de aceptar strings arbitrarios.

**Tests.**

```bash
ruff check .
mypy src/preprocess
mypy src/wisdom
pytest -q
lf validate experiments/dna_preprocess.yaml
```

Los tests offline cubren PDB/mmCIF/gzip, gramática, errores de modelo/cadena, filtros, altLoc, orden
explícito, plantillas, química peptídica/disulfuro/aromática, unión de relaciones, covalentes fuera de
radio, curvatura de esfera/plano/cilindro/concavidad, determinismo, pesos, integración
fuente→transformación→destino de LambdaForge, equivalencia de procesos CPU, fallos parciales,
invalidación científica de reanudación, identidad del artefacto dataset y debug acotado.

### 4.9. Limitaciones científicas

Estos límites determinan qué conclusiones pueden extraerse de la salida:

- Los negativos BTD-Combo son inferencias de benchmark muy filtradas, no prueba experimental de que
  una proteína jamás una DNA. QuickGO y BioLiP2 pueden ser incompletos o ir retrasados respecto a la
  biología conocida.
- Las API Data/Search de RCSB y el snapshot latest de BioLiP2 son servicios públicos vivos. Los bytes
  descargados y fijados son reproducibles y se cachean, pero una reconstrucción futura falla
  deliberadamente si cambia un snapshot en vez de atribuirle silenciosamente la versión antigua.
- Los grupos RCSB MMseqs2 al 30 % aíslan homología de forma conservadora, pero la identidad de
  secuencia no mide por completo relación estructural o evolutiva.
- Como explica 3.5, la nube aproxima SAS mediante esferas expandidas, no una SES analítica. No
  contiene parches cóncavos trazados por la superficie de la sonda, malla triangular ni toros.
- La separación firmada de 3.5 da signo interior/exterior y distancia exterior fiable. No se
  persiste como volumen ni se interpreta como distancia interior mínima exacta en solapes.
- La reducción voxel controla densidad de modo determinista. Blue-noise o Poisson-disk intentan además
  uniformar distancias entre vecinos; WISDOM no resuelve esa optimización.
- La curvatura es una aproximación cuadrática local de pendiente pequeña. Vecindarios escasos,
  ruidosos o desiguales pueden sesgarla; sustituir un resultado no finito por cero protege el archivo,
  no demuestra científicamente que la zona sea plana.
- Los pesos expresan espaciado local relativo y suman uno. No son áreas accesibles absolutas en Å² ni
  celdas Voronoi exactas, es decir, regiones más próximas a cada muestra.
- Las aristas aproximan localidad con distancia recta y normales. No calculan una **geodésica**, el
  camino más corto obligado a permanecer sobre la superficie; un paso estrecho difícil aún puede
  ganar un atajo o perder una conexión.
- Las plantillas cubren veinte aminoácidos estándar. La alternativa geométrica para otros residuos no
  sustituye una descripción completa del Chemical Component Dictionary para ligandos y modificaciones.
- Cargas y conexiones dependen de la fuente y Gemmi. WISDOM no añade átomos ausentes ni selecciona
  estados de protonación o tautomería —colocaciones alternativas de hidrógenos y dobles enlaces según
  las condiciones químicas— y no realiza cálculo cuántico.
- Las confianzas solo ordenan las fuentes de evidencia de 3.4. No son probabilidades calibradas
  experimentalmente de que exista un enlace.
- Solo se representa un modelo de coordenadas. Un ensemble de varios modelos o una trayectoria de
  dinámica molecular dependiente del tiempo necesitaría otra dimensión y no está soportado.

## 5. Modelos entrenables de WISDOM

WISDOMv1 responde una pregunta deliberadamente estrecha: ¿pueden la estructura atómica interna y la
geometría superficial fijas predecir una etiqueta binaria de la proteína completa y, a la vez,
exponer una puntuación por punto superficial? La etiqueta `0` o `1` pertenece a la proteína, no a un
átomo o punto. Por eso las puntuaciones locales tienen **supervisión débil**: solo aprenden mediante
la etiqueta global y no deben interpretarse como sitios de unión confirmados experimentalmente.

### 5.1. Manifiesto de etiquetas y batching de grafos

El preprocesado desconoce las etiquetas experimentales y la pertenencia a train/validation/test. El
entrenamiento las añade mediante el contrato CSV explícito mínimo:

```csv
file,label,split
processed/protein_1.npz,1,train
processed/protein_2.npz,0,val
processed/protein_3.npz,1,test
```

`file` es una ruta NPZ relativa al manifiesto, `label` vale exactamente `0` o `1` y `split` es
`train`, `val` o `test`. Los splits explícitos evitan filtrar una misma proteína entre entrenamiento
y evaluación. `WisdomDataset` abre cada NPZ con `allow_pickle=False`, comprueba arrays y rangos de
grafo necesarios y convierte solo esos arrays en tensores. No desplaza puntos, recalcula aristas ni
modifica el resultado del preprocesado.

Las proteínas tienen distinto número de átomos y puntos, de modo que no pueden apilarse como un
rectángulo sin padding. `WisdomCollator` construye una **unión disjunta**: concatena nodos y desplaza
cada extremo por el número de nodos anteriores. Los extremos atómicos reciben offsets atómicos, los
superficiales offsets superficiales y las dos filas del grafo bipartito superficie–átomo reciben su
offset correspondiente. `surface_batch[p]` registra a qué proteína pertenece el punto `p`. El
preprocesado guarda una sola vez cada arista no dirigida con `src<dst`; como la convolución envía
mensajes dirigidos, el collator añade determinísticamente `src→dst` y `dst→src`. Las máscaras
conservan su significado al convertirse en IDs R-GCN desde cero:

¿Por qué hace falta esta clase? Un batch de imágenes puede usar un tensor `[B,alto,ancho]` porque
todas comparten ejes rectangulares. Una proteína de 2.000 átomos no se puede apilar directamente con
otra de 700 y sus listas de aristas tampoco tienen igual longitud. Rellenar todos los grafos atómicos
y superficiales hasta la proteína mayor desperdiciaría memoria y crearía nodos falsos que cada
operación tendría que enmascarar. Las GNN de LambdaForge aceptan en cambio un grafo disperso; el
collator hace que varias proteínas parezcan un grafo grande, garantizando que ninguna arista cruce de
una proteína a otra.

Supongamos una proteína A con tres átomos y dos puntos superficiales, seguida de B con dos átomos y
tres puntos. Los índices locales de ambas empiezan en cero:

```text
                         proteína A       B antes del batch       B dentro del batch
índices atómicos         0, 1, 2          0, 1                    3, 4
índices superficiales    0, 1             0, 1, 2                 2, 3, 4
arista atómica           (0, 2)           (0, 1)                  (3, 4)
arista superficial       (0, 1)           (0, 2)                  (2, 4)
arista superficie→átomo  (1, 2)           (2, 1)                  (4, 4)
```

El offset atómico de B es tres y el superficial es dos. La arista bipartita necesita **offsets
distintos para sus filas**: la superficie `2` pasa a `4`, mientras el átomo `1` pasa a `4`. Aplicar
el mismo offset conectaría silenciosamente dominios incorrectos. Tras concatenar,
`surface_batch=[0,0,1,1,1]` indica que los dos primeros puntos pertenecen a A y los tres siguientes a
B; `atom_batch=[0,0,0,1,1]` registra la propiedad atómica equivalente. Los objetivos pasan a
`[y_A,y_B]`. WISDOMv1/v2 usan `surface_batch` para reducir predicciones locales a exactamente un
logit por proteína. El collator comprueba cada extremo desplazado para que un error falle de inmediato
en vez de mezclar proteínas durante el aprendizaje.

Por tanto, «collation» solo cambia contabilidad. No crea aristas científicas, recalcula distancias,
altera coordenadas ni permite fuga de información. Las filas quedan contiguas por eficiencia, pero
los grafos siguen siendo matemáticamente disjuntos.

| Máscara guardada | ID R-GCN | Significado |
|---:|---:|---|
| `1` | `0` | solo proximidad espacial |
| `2` | `1` | solo enlace covalente |
| `3` | `2` | espacial y covalente |

### 5.2. Modelos, ecuaciones y formas tensoriales de WISDOMv1

`WisdomV1` es la única composición neuronal específica del dominio en v1. No reimplementa
aprendizaje de grafos: su constructor crea `RelationalGCN`, `MLP`, `GCN`, scatter indexado y pooling
disperso de LambdaForge a partir de parámetros conceptuales independientes. Así el HPO puede cambiar
una anchura o profundidad sin dejar otro `in_channels` obsoleto dentro del YAML.

| Componente | Implementación | Entrada → salida | Qué aprende |
|---|---|---|---|
| Embedding de elemento | `torch.nn.Embedding` | número atómico `[N]` → `[N,E]` | Un vector aprendido por ID de elemento químico. |
| Embedding opcional de residuo | `torch.nn.Embedding` | ID de residuo `[N]` → `[N,E]` | Comprueba si la categoría de aminoácido aporta contexto útil. |
| Encoder atómico | LambdaForge `RelationalGCN` | features `[N,E]` o `[N,2E]`, aristas y relaciones → `[N,D]` | Matrices distintas para aristas espaciales, covalentes y combinadas. |
| Transferencia átomo→superficie | LambdaForge `Scatter.mean` | embeddings e incidencias → `[M,D]` | Contexto atómico medio asociado a cada punto. |
| Proyección superficial | LambdaForge `MLP` | contexto y curvatura `[M,D+3S]` → `[M,D]` | Fusiona ambas fuentes punto a punto. |
| Encoder superficial | LambdaForge `GCN` | features `[M,D]` y aristas → `[M,D]` | Intercambia información con vecinos superficiales. |
| Head local | `torch.nn.Linear(D,1)` | embedding superficial `[M,D]` → logits `[M]` | Produce evidencia local de clase. |
| Reducción global | LambdaForge `SparseMaxPooling` | logits y `surface_batch` → `[B]` | Implementa la regla existencial MAX MIL fija. |

Un embedding es una tabla de consulta entrenable, no un descriptor químico escrito a mano. R-GCN
significa **red convolucional relacional de grafos**: un vecino unido por enlace covalente se
transforma de forma distinta a otro unido solo por proximidad. El MLP posterior es un perceptrón
multicapa aplicado fila a fila; en esta configuración es una única proyección aprendida. El GCN
superficial permite después que cada punto combine su estado con los recibidos por el grafo. Dos
componentes de ese grafo nunca intercambian mensajes GCN, tal como explica 3.6.

Sean `N` los átomos totales, `M` los puntos superficiales, `B` las proteínas, `E` la anchura del
embedding, `D` la anchura oculta y `S` las escalas de curvatura. La tabla de residuos se omite por
completo en el candidato que usa solo elemento; en el otro se concatenan ambos embeddings. El
`RelationalGCN` emplea las tres relaciones para producir `h_atom[N,D]`.

Para el punto `p`, sea `A(p)` el conjunto de átomos conectados mediante el grafo bipartito. La
primera transferencia átomo–superficie es intencionadamente una media:

```math
h_{A\to S}(p)=\frac{1}{|A(p)|}\sum_{a\in A(p)}h_{atom}(a).
```

El preprocesado garantiza al menos un átomo asociado por punto y el scatter indexado de LambdaForge
calcula la expresión sin una matriz densa átomo-por-punto. Cada punto posee además `S` tripletes
`[H,K,C]`; aplanarlos produce `3S` escalares invariantes. WISDOM concatena esos valores con
`h_A→S`, proyecta el tensor `[M,D+3S]` mediante un `MLP` de LambdaForge y lo pasa por un `GCN` de dos
capas sobre el grafo superficial. Se excluyen posiciones absolutas y componentes de las normales;
rotar toda la entrada no puede cambiar una feature solo porque haya cambiado un eje cartesiano.

Una capa lineal convierte cada embedding superficial en un logit local `l_p`. Un «logit» es un
número real previo a sigmoid: positivo favorece clase `1`, negativo clase `0` y cero equivale a
probabilidad `0,5`. Para la proteína `b`, sea `P_b` su conjunto de puntos. El logit v1 es MAX:

```math
L_b=\max_{p\in P_b}l_p.
```

Esto expresa «la proteína puede ser positiva si existe al menos un punto con evidencia positiva
fuerte». También puede sobreajustar a un único punto espurio, riesgo que v2 aísla deliberadamente.
El modelo devuelve `logits[B]` y `surface_logits[M]`; solo el primero tiene etiqueta verdadera, por
lo que una puntuación local es evidencia y no una anotación funcional validada.

Para objetivo `y_b∈{0,1}`, la entropía cruzada binaria con logits de LambdaForge minimiza

```math
\mathcal L_b=-y_b\log\sigma(L_b)-(1-y_b)\log(1-\sigma(L_b)),
```

donde `σ(z)=1/(1+e^{-z})` transforma el logit en probabilidad. AUROC mide con qué frecuencia un
positivo aleatorio queda por encima de un negativo a través de todos los umbrales. AUPRC resume
precisión frente a recall y resulta especialmente informativa cuando hay pocos positivos.

Ninguno de estos nombres implica actualizar coordenadas tridimensionales. Las posiciones del
preprocesado determinan los grafos dispersos, pero v1 no entrega coordenadas cartesianas ni normales
a las capas neuronales. Usa geometría mediante curvaturas invariantes y topología, evitando depender
de una rotación global arbitraria.

V1 busca solo elecciones fundamentales: features de elemento o elemento más residuo;
`E∈{16,32,64}`; `D∈{64,128,256}`; de una a cuatro capas R-GCN; de una a tres capas de proyección; de
una a cuatro capas GCN superficiales; dropout compartido en `[0,0,5]`; weight decay entre `10^-6` y
`10^-3`; y learning rate entre `10^-5` y `3×10^-3`. La media átomo→superficie, MAX global,
preprocesado, grafos y splits quedan fijos para responder una sola pregunta.

### 5.3. Pooling y diagnósticos de localización de WISDOMv2

WISDOMv2 pregunta si una regla distinta de MAX puede conservar la clasificación y reducir la
dependencia de un único punto extremo accidental. Parte del backbone v1 revisado y materializado
explícitamente y solo cambia la operación que transforma logits locales en uno de proteína.
Features, embeddings, R-GCN, media átomo→superficie, proyección, GCN superficial y head local quedan
controlados. V2 no vuelve a buscar sus anchuras ni profundidades.

MAX y attention usan poolings dispersos de LambdaForge; la media ponderada por área usa su reducción
`Scatter`. Top-k y log-sum-exp compactan solo logits escalares en `X[B,N_max,1]`; una máscara excluye
el padding. Los grafos atómico y superficial siguen dispersos y no se crean aristas falsas.

La atención global usa `SparseAttentionPooling` de LambdaForge. Sea `h_p∈R^D` la representación
aprendida del punto `p` y `l_p` su logit de positividad independiente. Attention calcula

```math
s_p=\mathbf v^\top\tanh(\mathbf V h_p),
\qquad
\alpha_p=\frac{e^{s_p}}{\sum_{q\in P_b}e^{s_q}},
\qquad
L_b=\sum_{p\in P_b}\alpha_p l_p.
```

`V` proyecta la representación y `v` produce una puntuación. Los pesos `α_p` son positivos y suman
uno dentro de cada proteína. Significan
«importancia para esta decisión de bag», no la positividad local `l_p`, y no deben presentarse
automáticamente como explicación de un sitio funcional.

La interfaz controlada de v2 compara estas reglas:

| Valor YAML | Implementación | Logit de proteína y comportamiento buscado |
|---|---|---|
| `max` | `SparseMaxPooling` | Control existencial v1 exacto: `L_b=max_p l_p`. |
| `mean` | `Scatter.sum` de LambdaForge | Media por área: `L_b=sum_p w_p l_p/sum_p w_p`. |
| `attention` | `SparseAttentionPooling` | Importancia aprendida de `h_p`, aplicada a logits de positividad. |
| `topk` | `FractionalTopKMeanPooling` | Media de los `ceil(f|P_b|)` logits mayores, con `f` entre 1 % y 20 %. |
| `local_mean_max` | Consenso regional WISDOM más `SparseMaxPooling` | Media local ponderada por área en el grafo existente, seguida de MAX global. |
| `log_sum_exp` | LambdaForge `LogSumExpPooling` normalizado | `L_b=β^-1 log(|P_b|^-1 sum_p exp(βl_p))`, control de máximo suave. |

Para la hipótesis regional principal, sea `N(j)` el conjunto de vértices con una arista dirigida
hacia `j`; sea `w_i>0` el área representada; y sea `r_i^(0)=l_i`. El nivel de consenso siguiente es

```math
r_j^{(t+1)}=
\frac{w_j r_j^{(t)}+\sum_{i\in N(j)}w_i r_i^{(t)}}
     {w_j+\sum_{i\in N(j)}w_i},
\qquad
L_b=\max_{j\in P_b}r_j^{(T)}.
```

El numerador combina evidencia ponderada por área del punto y sus vecinos; el denominador es el
área representada total. `T∈{1,2,3}` amplía la región un salto del grafo por nivel. Un pico aislado
se diluye, mientras un parche positivo coherente permanece positivo. No se reconstruye geometría.
Los tests matemáticos cubren picos aislados frente a regiones, pesos desiguales y separación entre
proteínas. La atención local se aplaza: LambdaForge aporta atención global de conjuntos, pero añadir
ahora un operador aprendido sobre vecindarios variables confundiría esta primera prueba de consenso.

Log-sum-exp resta internamente su máximo por estabilidad y normaliza por número de puntos. Top-k
fraccional siempre toma al menos uno. Noisy-OR sigue ausente porque considerar miles de puntos como
Bernoulli independientes satura `1-product(1-p)` cerca de uno sin un modelo físico que lo justifique.

V2 expone mapas guardables en el orden original de puntos del NPZ:

- `surface_logits[M]` y `surface_probabilities[M]=sigmoid(surface_logits)`;
- `localization_scores[M]`, distribución con área
  `q_p = w_p exp(l_p) / sum_q(w_q exp(l_q))`, normalizada por proteína;
- `positive_area_fraction[B]`, área representada normalizada con probabilidad local al menos 0,5;
- `maximum_surface_probability[B]`;
- `localization_entropy[B]`, igual a `-sum(q_p log q_p)/log(|P_b|)` con más de un punto. Cerca de
  cero significa concentración y cerca de uno un mapa difuso.

Estos diagnósticos describen el mapa del modelo; no son etiquetas locales ni se añaden a la loss.
`localization_scores` ofrece una escala común, no necesariamente el peso interno exacto de cada
pooling. Estos diagnósticos del entrenamiento no consumen etiquetas puntuales. El evaluador post-run
separado contrasta el mapa con sidecars DNA inmutables tras seleccionar el modelo; esa comparación
posterior nunca modifica loss ni el objetivo HPO.

### 5.4. Entrenamiento, evaluación post-run y artefactos 3D

WISDOM no contiene scheduler de HPO, bucle de pruning, gestor de semillas, selector de GPU ni base
de datos de estudios. LambdaForge controla materialización, inicialización Sobol, búsqueda bayesiana,
knowledge gradient consciente del coste, fidelidad por épocas acumuladas, curvas de aprendizaje,
pruning, carrera adaptativa de semillas, confirmación, checkpoints, admisión de memoria, persistencia,
reanudación, agregación, informes e indexado. WISDOM aporta dataset, collator, modelos, espacios
científicos y la semántica regional.

| Configuración | Responsabilidad |
|---|---|
| `wisdom_v1.yaml` | Modelo, entrenamiento y HPO adaptativo completos de v1, con media átomo→superficie y MAX fijos. |
| `wisdom_v2.yaml` | Modelo v2 completo, valores del backbone v1 seleccionado y HPO condicional exclusivamente de pooling. |

V1 optimiza AUPRC de validación, nunca test. Sobol con `trials:auto` deriva el diseño inicial de la
dimensión efectiva; después actúa la búsqueda bayesiana con knowledge gradient consciente del coste.
Cada candidato empieza con cinco épocas y puede reanudarse en incrementos de cinco hasta 100. El
pruning posterior comienza tras diez. Las semillas `[7,17,27]` compiten adaptativamente y tres
finalistas usan semillas nuevas `[101,211]`. Los límites duros son 60 acciones, 2.000 épocas y
200.000 segundos GPU: limitan gasto, no obligan a agotarlo.

V2 usa un único HPO condicional. Compara MAX, mean, attention, cinco fracciones top-k, de uno a tres
niveles regionales y log-sum-exp. La fracción solo existe para `topk`, los niveles para
`local_mean_max`, la anchura de atención para `attention` y beta para log-sum-exp. Features, anchuras,
profundidades, dropout, learning rate y weight decay están ausentes: se copian de la selección robusta
de v1 y permanecen como constantes controladas.

Los tres loaders usan `{dataset: wisdom-dna, version: "2", subpath: manifest.csv}`, no una ruta
absoluta de máquina. LambdaForge resuelve `wisdom-dna@2` mediante DatasetRegistry y registra en la
evidencia materializada identidad exacta de contenido/build y placement elegido. Un equipo local y
un clúster pueden guardar copias verificadas en rutas distintas sin editar parámetros ni cambiar la
identidad científica. Construye o materializa la versión antes del HPO; la ausencia de datos nunca
se convierte silenciosamente en split aleatorio ni etiquetas sintéticas.

En un clúster gestionado, primero se garantiza que ese clúster tenga un placement verificado y
después se lanza allí el experimento. No se pasa ninguna ruta al comando de entrenamiento porque el
selector lógico ya está en el YAML:

```bash
lf datasets materialize wisdom-dna@2 --on citius-ctgpgpu12 --strategy replicate --apply
lf run experiments/wisdom_v1.yaml --on citius-ctgpgpu12
```

Inspecciona composición y planes sin crear estado de estudio:

```bash
lf datasets list --all
lf datasets show wisdom-dna@2
lf datasets locations wisdom-dna@2
lf validate experiments/wisdom_v1.yaml
lf compose experiments/wisdom_v1.yaml
lf inspect experiments/wisdom_v1.yaml --resolved
lf run experiments/wisdom_v1.yaml --dry-run

lf validate experiments/wisdom_v2.yaml
lf inspect experiments/wisdom_v2.yaml --resolved
lf run experiments/wisdom_v2.yaml --dry-run
```

Instala el extra `adaptive-hpo` de LambdaForge para la búsqueda bayesiana. El comando normal inicia
v1; repetirlo reanuda el estudio duradero desde `.lambdaforge/adaptive/<study-id>/`. `--restart`
descarta explícitamente el progreso. No edites a mano `state.json`, los eventos append-only ni
`summary.json`.

```bash
lambdaforge run experiments/wisdom_v1.yaml
lambdaforge run experiments/wisdom_v1.yaml
lambdaforge results audit experiments/wisdom_v1.yaml --no-archived --write-index \
  --fail-on-ambiguous
lambdaforge aggregate experiments/wisdom_v1.yaml
```

Revisa medias y dispersión de confirmación, curvas, límites sospechosos y simplicidad; no copies el
mayor decimal sin más. Copia entonces los valores de backbone y optimizador seleccionados en los
bloques fijos claramente indicados de `wisdom_v2.yaml` y ejecuta su HPO de pooling:

```bash
lambdaforge run experiments/wisdom_v2.yaml
lambdaforge aggregate experiments/wisdom_v2.yaml
```

El YAML v2 conserva una advertencia en `scientific_status` hasta completar esa copia manual. El
equipo comprobado tiene una GPU de 4 GB, por eso la concurrencia HPO es uno y LambdaForge reserva 3
GiB más margen con límite defensivo del asignador. Son ajustes operativos, no científicos; un destino
gestionado puede sustituir recursos sin cambiar la identidad del candidato. La inspección genérica
de arrays NPZ y visualización 3D con roles explícitos de LambdaForge siguen documentadas en 4.2.
WISDOM solo conserva validación que una herramienta genérica no puede inferir: topología proteica,
gaps con signo, orientación de normales e identidades de curvatura.

El análisis DNA final es una acción `post_run` obligatoria de LambdaForge que se ejecuta en la misma
asignación de recursos después de confirmar correctamente el entrenamiento. En estos YAML
adaptativos, `scope: confirmed_runs` restringe las costosas métricas de test y exportaciones 3D a las
configuraciones finalistas reentrenadas con semillas de confirmación independientes. En un
experimento no adaptativo, la misma acción se ejecutaría tras una finalización normal o un early
stopping del entrenador. Los trials de búsqueda pausados, descartados por pruning, cancelados,
fallidos o interrumpidos cooperativamente no ejecutan acciones post-run. `scope: all_runs` puede
incluir todos los runs terminales correctos de la búsqueda, pero no convierte ninguno de esos
estados excluidos en un estado post-run.

El YAML solicita explícitamente `checkpoint: best`; LambdaForge no lo sustituye silenciosamente por
`last`. Si no existe un checkpoint best inequívoco, la acción obligatoria falla de forma visible,
mientras el checkpoint de entrenamiento ya confirmado queda disponible para diagnosticar y
reintentar solo el post-run. La acción usa el SHA-256 del checkpoint persistido, su propia identidad
de configuración y los hashes de los artefactos declarados para escribir un recibo duradero. Al
repetir un run idéntico terminado se reutiliza un recibo verificado; cambiar la política de
evaluación o perder un artefacto repite solo la acción afectada, no el entrenamiento neuronal. El
ground truth superficial nunca entra en loss, gradientes, AUPRC de validación, política HPO ni
selección de modelo.

El informe global usa accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, kappa de
Cohen, AUROC y AUPRC de LambdaForge. El superficial aplica la misma familia a puntos con
`surface_valid_mask` verdadero. Las probabilidades son
(\sigma(\ell_i)=1/(1+e^{-\ell_i})), con logit local (ell_i) y threshold discreto 0.5. Micro
concatena puntos válidos; macro calcula primero por proteína y promedia solo valores definidos. El
informe separa el resumen global de la localización primaria sobre positivos y recalcula métricas
positivas para cada umbral de sensibilidad DNA guardado, sin reentrenar. Se conservan conteos de
proteínas y métricas indefinidas: denominador inválido es `null`, nunca cero.
Los positivos añaden Dice e IoU; los negativos curados informan fracción de área positiva predicha,
probabilidad máxima, número de hotspots espurios conectados y tamaño del mayor.

Los artefactos post-run permanecen dentro del run de entrenamiento:

```text
evaluation/
├── metrics/global_metrics.json
├── metrics/surface_metrics.json
├── metrics/per_protein_metrics.csv
├── predictions/global_predictions.csv
├── visualizations/<protein>.ply
├── visualizations/<protein>.npz
└── report/evaluation_summary.{json,md}
```

PLY es un formato estándar de nube de puntos. Cada vértice conserva el mismo `(x,y,z)` y puede
exponer normales, área, curvaturas, target hard/soft, validez, distancia al DNA, logit y probabilidad
como escalares seleccionables. El NPZ compañero mantiene esos canales sin pérdida textual. Las
dimensiones aprendidas se llaman `latent_chemical_channel_N`, no carga electrostática ni otra
propiedad física que no representen. Solo se exportan índices explícitos; no se aplica PCA
silenciosamente y la visualización nunca cambia valores usados por las métricas.

V1 y v2 omiten distancias de arista atómica, coordenadas absolutas, normales como features
neuronales, heads de residuo, kernels cuasi-geodésicos, actualizaciones equivariantes de coordenadas,
convoluciones dMaSIF, rondas bidireccionales átomo↔superficie, aprendizaje contrastivo, modelos de
lenguaje y salidas multitarea. V3–v7 siguen solo documentadas en
[`docs/model_roadmap.md`](docs/model_roadmap.md). V2 es técnicamente ejecutable, pero no debe
describirse como mejor hasta comparar los poolings declarados con etiquetas reales, semillas
pareadas y confirmación independiente.

## 6. Bibliografía

1. Berman, H. M. et al. (2000). “The Protein Data Bank.” *Nucleic Acids Research*, 28(1),
   235–242. [doi:10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235).
2. Bourne, P. E. et al. (1997). “Macromolecular Crystallographic Information File.” *Methods in
   Enzymology*, 277, 571–590.
   [doi:10.1016/S0076-6879(97)77032-0](https://doi.org/10.1016/S0076-6879(97)77032-0).
3. Wojdyr, M. (2022). “GEMMI: A library for structural biology.” *Journal of Open Source
   Software*, 7(73), 4200. [doi:10.21105/joss.04200](https://doi.org/10.21105/joss.04200).
4. Lee, B. & Richards, F. M. (1971). “The interpretation of protein structures: Estimation of
   static accessibility.” *Journal of Molecular Biology*, 55(3), 379–400.
   [doi:10.1016/0022-2836(71)90324-X](https://doi.org/10.1016/0022-2836(71)90324-X).
5. Shrake, A. & Rupley, J. A. (1973). “Environment and exposure to solvent of protein atoms:
   Lysozyme and insulin.” *Journal of Molecular Biology*, 79(2), 351–371.
   [doi:10.1016/0022-2836(73)90011-9](https://doi.org/10.1016/0022-2836(73)90011-9).
6. Bondi, A. (1964). “van der Waals Volumes and Radii.” *Journal of Physical Chemistry*, 68(3),
   441–451. [doi:10.1021/j100785a001](https://doi.org/10.1021/j100785a001).
7. Cordero, B. et al. (2008). “Covalent radii revisited.” *Dalton Transactions*, 21, 2832–2838.
   [doi:10.1039/B801115J](https://doi.org/10.1039/B801115J).
8. Saff, E. B. & Kuijlaars, A. B. J. (1997). “Distributing many points on a sphere.” *The
   Mathematical Intelligencer*, 19, 5–11.
   [doi:10.1007/BF03024331](https://doi.org/10.1007/BF03024331).
9. Cazals, F. & Pouget, M. (2005). “Estimating differential quantities using polynomial fitting of
   osculating jets.” *Computer Aided Geometric Design*, 22(2), 121–146.
   [doi:10.1016/j.cagd.2004.09.004](https://doi.org/10.1016/j.cagd.2004.09.004).
10. Bentley, J. L. (1975). “Multidimensional binary search trees used for associative searching.”
    *Communications of the ACM*, 18(9), 509–517.
    [doi:10.1145/361002.361007](https://doi.org/10.1145/361002.361007).
11. Sverrisson, F., Feydy, J., Correia, B. E. & Bronstein, M. M. (2021). “Fast End-to-End Learning
    on Protein Surfaces.” *CVPR 2021*, 15272–15281.
    [Artículo open access](https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html).
12. Gainza, P. et al. (2020). “Deciphering interaction fingerprints from protein molecular surfaces
    using geometric deep learning.” *Nature Methods*, 17, 184–192.
    [doi:10.1038/s41592-019-0666-6](https://doi.org/10.1038/s41592-019-0666-6).
13. Kipf, T. N. & Welling, M. (2017). “Semi-Supervised Classification with Graph Convolutional
    Networks.” *ICLR 2017*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907).
14. Schlichtkrull, M. et al. (2018). “Modeling Relational Data with Graph Convolutional Networks.”
    *ESWC 2018*, 593–607. [doi:10.1007/978-3-319-93417-4_38](https://doi.org/10.1007/978-3-319-93417-4_38).
15. Loshchilov, I. & Hutter, F. (2019). “Decoupled Weight Decay Regularization.” *ICLR 2019*.
    [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
16. Ilse, M., Tomczak, J. M. & Welling, M. (2018). “Attention-based Deep Multiple Instance
    Learning.” *Proceedings of Machine Learning Research*, 80, 2127–2136.
    [Artículo en PMLR](https://proceedings.mlr.press/v80/ilse18a.html).
17. Burley, S. K. et al. (2023). “RCSB Protein Data Bank (RCSB.org): delivery of experimentally
    determined PDB structures alongside one million computed structure models of proteins from
    artificial intelligence/machine learning.” *Nucleic Acids Research*, 51(D1), D488–D508.
    [doi:10.1093/nar/gkac1077](https://doi.org/10.1093/nar/gkac1077).
18. Steinegger, M. & Söding, J. (2017). “MMseqs2 enables sensitive protein sequence searching for
    the analysis of massive data sets.” *Nature Biotechnology*, 35, 1026–1028.
    [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988).
19. Binns, D. et al. (2009). “QuickGO: a web-based tool for Gene Ontology searching.”
    *Bioinformatics*, 25(22), 3045–3046.
    [doi:10.1093/bioinformatics/btp536](https://doi.org/10.1093/bioinformatics/btp536).
20. Luscombe, N. M., Laskowski, R. A. & Thornton, J. M. (2001). “Amino acid–base interactions: a
    three-dimensional analysis of protein–DNA interactions at an atomic level.” *Nucleic Acids
    Research*, 29(13), 2860–2874.
    [doi:10.1093/nar/29.13.2860](https://doi.org/10.1093/nar/29.13.2860).
21. Li, P., Liu, Y., Liang, L. & Liu, R. (2026). “Datasets for DyProL: Conformational Ensembles of
    Nucleic Acid-Binding Proteins,” versión 1. Zenodo.
    [doi:10.5281/zenodo.19547616](https://doi.org/10.5281/zenodo.19547616).
22. Rahman, C. R. et al. (2025). “Benchmarking recent computational tools for DNA-binding protein
    identification.” *Briefings in Bioinformatics*, 26(1), bbae634.
    [doi:10.1093/bib/bbae634](https://doi.org/10.1093/bib/bbae634).
23. Zhang, C., Zhang, X., Freddolino, L. & Zhang, Y. (2024). “BioLiP2: an updated structure database
    for biologically relevant ligand–protein interactions.” *Nucleic Acids Research*, 52(D1),
    D404–D412. [doi:10.1093/nar/gkad630](https://doi.org/10.1093/nar/gkad630).

La superficie de WISDOM se implementó independientemente. dMaSIF y MaSIF motivan el futuro uso de
representaciones superficiales aprendidas, pero WISDOM no copia su código ni afirma identidad
algorítmica con ninguno de ellos.
