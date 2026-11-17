# WISDOM — aprendizaje sobre superficies de proteínas

[English](README.md)

WISDOM aprende una puntuación en cada punto de la superficie de una proteína usando etiquetas de
la **proteína completa**. El primer benchmark pregunta si una proteína se une al ADN. El objetivo
no es solo acertar sí/no, sino obtener un mapa útil de la posible región de unión. Son objetivos
distintos: clasificar bien una proteína no demuestra que se localice bien su región de unión.

Este README explica instalación, archivos, ejecución y organización del código. La
[memoria científica](docs/WISDOM-memoria.pdf), actualmente en español, contiene las derivaciones,
el razonamiento experimental, los resultados y las limitaciones. Distingue capacidades implementadas,
propuestas y hallazgos validados. Los comentarios de los YAML documentan las decisiones de cada estudio.

## 0. Índice

- [1. Inicio rápido](#1-inicio-rápido)
- [2. Instalación](#2-instalación)
  - [2.1. Entornos locales y del clúster](#21-entornos-locales-y-del-clúster)
  - [2.2. Actualizar la memoria científica](#22-actualizar-la-memoria-científica)
- [3. Construir e inspeccionar datasets](#3-construir-e-inspeccionar-datasets)
  - [3.1. Benchmark de unión a ADN](#31-benchmark-de-unión-a-adn)
  - [3.2. Reutilización, versiones y zinc](#32-reutilización-versiones-y-zinc)
- [4. Contratos técnicos](#4-contratos-técnicos)
  - [4.1. Arrays estructurales y anotaciones](#41-arrays-estructurales-y-anotaciones)
  - [4.2. Modelos, supervisión y métricas](#42-modelos-supervisión-y-métricas)
  - [4.3. Controles radiales y de anchura de SchNet](#43-controles-radiales-y-de-anchura-de-schnet)
  - [4.4. Selección de fuentes por grupos](#44-selección-de-fuentes-por-grupos)
  - [4.5. Auditorías geométricas congeladas y entradas multiescala](#45-auditorías-geométricas-congeladas-y-entradas-multiescala)
  - [4.6. Consolidación y supervisión de regiones negativas](#46-consolidación-y-supervisión-de-regiones-negativas)
  - [4.7. Targets físicos VF](#47-targets-físicos-vf)
- [5. Experimentos y desarrollo](#5-experimentos-y-desarrollo)
  - [5.1. Entrenamiento y visores de proteínas](#51-entrenamiento-y-visores-de-proteínas)
  - [5.2. Organización del código y verificación](#52-organización-del-código-y-verificación)
- [6. Referencias](#6-referencias)

## 1. Inicio rápido

Desde la raíz del repositorio:

```bash
./install.sh
conda activate wisdom
lf validate experiments/preprocess/dna/dna_preprocess.yaml
lf explain experiments/preprocess/dna/dna_preprocess.yaml
lf run experiments/preprocess/dna/dna_preprocess.yaml --dry-run
```

La validación y la simulación planifican el trabajo; no construyen un dataset ni entrenan un modelo.
El YAML de ADN actual reutiliza `data/dna/design`, construye la vista de entrenamiento del 25% y
conserva los splits de evaluación configurados. Incluye el paquete opcional de modelos de `data/models`.
Revisa esas entradas y el nombre/versión del dataset antes de lanzarlo. El §3 explica la reutilización
y el §4 los archivos resultantes.

Cuando esté preparado, el investigador lo lanza explícitamente:

```bash
lf run experiments/preprocess/dna/dna_preprocess.yaml --on YOUR_CLUSTER
lf overview
lf logs wisdom-dna-preprocess --follow
```

Para entrenar, elige un estudio del [catálogo de experimentos](experiments/README.md), comprueba que
su dataset esté registrado en el destino y valida y lanza ese YAML. Los números de experimento
describen etapas científicas, no la generación del modelo Python.

## 2. Instalación

### 2.1. Entornos locales y del clúster

Usa `./install.sh`, no un entorno virtual de Python independiente. El script reutiliza Conda o
propone instalar Miniforge, prepara el entorno `wisdom` e instala WISDOM y LambdaForge en modo
editable. Reutiliza un repositorio local de LambdaForge si está disponible. `./install.sh --yes`
acepta la instalación predeterminada. Las dependencias están en `pyproject.toml` y `environment.yml`;
actualmente se exige LambdaForge >=0.19.0 sin límite superior de versión.

Conda también proporciona **MMseqs2** y **Foldseek**, los programas que identifican similitudes de
secuencia y estructura durante la selección. **Gemmi** lee las coordenadas depositadas. PyTorch
ejecuta los modelos entrenables; CUDA permite usar GPU NVIDIA compatibles. No se necesita GPU para
comprobar configuraciones ni para ejecutar las pruebas unitarias de CPU.

```bash
conda activate wisdom
python -m pip check
command -v mmseqs
command -v foldseek
lf doctor
```

El entorno remoto pertenece al proyecto, no a una ruta fija de Conda. Prepara un clúster configurado
con `lf clusters bootstrap YOUR_CLUSTER --project .`; usa el contrato de Conda de
`[tool.lambdaforge.environment]`. Repítelo tras cambios relevantes en las dependencias. Para actualizar
el entorno local, vuelve a ejecutar `./install.sh`. Los pesos externos de proteínas son entradas
independientes declaradas, no descargas ocultas del entrenamiento. Su inventario se explica en
[data/models/README.md](data/models/README.md).

### 2.2. Actualizar la memoria científica

`docs/WISDOM-memoria.pdf` es una **copia real**, no un enlace al ordenador del autor. El JSON adjunto
registra el nombre del archivo, su huella SHA-256 de contenido y su tamaño. La memoria se mantiene
fuera del repositorio; WISDOM no reescribe sus resultados ni puede saber por sí solo cuándo cambia.
Después de compilarla, sincroniza y comprueba la copia antes de hacer commit:

```bash
python scripts/sync_report.py /path/to/Memoria/main.pdf
python scripts/sync_report.py /path/to/Memoria/main.pdf --check
```

La comprobación falla si los bytes o los metadatos están desactualizados. Las fechas no se usan como
prueba de igualdad. Mantén aquí las instrucciones de uso, en los YAML la configuración experimental
y en la memoria las explicaciones científicas extensas; no añadas un diario Markdown por cada fallo
de ejecución. Las mediciones locales de rendimiento van en `benchmarks/`, no en las conclusiones.

## 3. Construir e inspeccionar datasets

### 3.1. Benchmark de unión a ADN

Una estructura del **Protein Data Bank**, consultada mediante **RCSB**, describe coordenadas
observadas bajo unas condiciones experimentales concretas. Que no contenga ADN **no** demuestra
que la proteína no pueda unirse a él. Por eso WISDOM exige evidencia negativa curada explícita.
BTD y BTD-Combo son las fuentes de benchmark de unión usadas al preparar la evidencia; se conserva
la etiqueta junto con su evidencia y limitaciones, no como prueba universal de incapacidad de unión.

| Etapa | Entrada | Trabajo realizado | Resultado para la siguiente etapa |
|---|---|---|---|
| Preparación de fuentes | Registros del benchmark y estructuras depositadas | Resolver evidencia y comprobar si puede obtenerse referencia local | Evidencia tipada `raw.jsonl` |
| Selección | Evidencia congelada | Verificar contactos con ADN, agrupar proteínas relacionadas, caracterizar formas, balancear y dividir | `design/` portable |
| Preprocesado | Design seleccionado exacto | Construir arrays estructurales y anotaciones de ADN separadas; validar y publicar | Dataset inmutable registrado |
| Visualización | Dataset registrado | Representar las entradas estructurales para inspeccionarlas | Galería HTML |

`scripts/create_fasta.py` prepara la evidencia; pese a su nombre histórico, la entrada de selección
es JSONL tipado, un registro JSON por línea. Consulta `python scripts/create_fasta.py --help` para
sus opciones de adquisición. Una cadena positiva debe contactar con ADN en el **ensamblaje biológico**
seleccionado: el complejo molecular interpretado como relevante biológicamente, no vecinos arbitrarios
del cristal. La evidencia contradictoria se aparta para revisarla, no se reetiqueta en silencio.

La selección usa MMseqs2/Foldseek y conexiones de identidad para formar **grupos de leakage**:
miembros relacionados deben permanecer en el mismo split para que la evaluación no se beneficie de
ejemplos casi duplicados del entrenamiento. Por separado, HDBSCAN agrupa formas físicas en
**fenotipos**. En positivos describen la forma local del sitio de unión; en negativos, la morfología
global. Sirven para cubrir diversidad y auditar, no para definir homología o función. Los elementos
sin clúster se mantienen explícitos.

Solo después de formar esos grupos se fijan train/validation/test y las **diluciones** anidadas:
subconjuntos más pequeños de la población de entrenamiento original. Validación y test no se diluyen.
Los grupos indivisibles pueden impedir un balance exacto; consulta `REPORT.md` y las estadísticas
generadas en lugar de suponer que todos los splits son exactamente 50/50.

El design portable contiene TXT etiquetados, `catalog.csv`, pertenencia a diluciones, manifiestos,
la copia de coordenadas seleccionadas y su índice. Una línea etiquetada es `1ABC_AQ<TAB>1`: una cadena
llamada `AQ`. `1ABC_A_Q` indica las cadenas `A` y `Q`. Las etiquetas son `0` o `1`; los splits son
explícitos. Una lista de identificadores sin etiqueta no permite reconstruir ensamblajes y contactos.

### 3.2. Reutilización, versiones y zinc

El YAML de ADN tiene tres pasos que pueden omitirse independientemente: `select`, `preprocess` y
`visualize`. Estos son los modos admitidos; modifica las recetas comentadas del YAML, no los algoritmos:

| Intención | Selección | Preprocesado | Visualización |
|---|---|---|---|
| Reconstruir desde evidencia congelada | `skip: false`, indicar `raw_path` | `skip: false` | Opcional |
| Reutilizar un design auditado | `skip: true`, indicar `existing_design` | `skip: false` | Opcional |
| Inspeccionar un dataset existente | Omitir, sin entrada de design | `skip: true` | Indicar un selector de dataset registrado |

`{from: select.train}` referencia la salida llamada `train` que declara Selection, no un nombre de
archivo adivinado por LambdaForge. El paso siguiente recibe su ruta preparada. Igualmente,
`{from: preprocess.dataset}` resuelve el dataset creado en este flujo; no lo sustituyas por una
versión que aún no existe. Reutilizar un design requiere toda su copia de estructuras, no solo
`train-labelled.txt`.

Los datasets gestionados son inmutables y se registran por nombre/versión. La receta de ADN actual
publica `wisdom-dna-reduced@8`; los estudios anteriores pueden usar deliberadamente
`wisdom-dna-reduced@6`. No sustituyas uno por otro en silencio. Un dataset construido solo con
train-25 usa `subset: full` al entrenar; `replicate-00/train-25` corresponde únicamente a una versión
que publique esa vista adicional. Las familias estructurales nuevas son archivos independientes
ligados al contenido; enriquecer un dataset no garantiza equivalencia con reconstruir cambiando geometría.

```bash
lf datasets list
lf datasets show wisdom-dna-reduced@6
lf datasets stats wisdom-dna-reduced@6
lf datasets verify wisdom-dna-reduced@6
lf datasets materialize wisdom-dna-reduced@6 --on YOUR_CLUSTER --strategy replicate --apply
lf validate experiments/preprocess/dna/validate_dna.yaml
```

El último YAML realiza una auditoría científica independiente de los checks de integridad de LF.
Los reintentos compatibles reutilizan checkpoints validados por proteína; las fechas por sí solas
no autorizan reutilización ni invalidan bytes iguales. Adoptar una revisión nueva de RCSB requiere
una selección nueva explícita, no reemplazar las coordenadas dentro de un design auditado.

El enriquecimiento guarda un checkpoint nativo validado por proteína y familia generada antes
de publicar el dataset completo. Unos pocos fallos no borran los archivos correctos, pero la
versión de destino no se publica hasta que todos los miembros terminan bien. La recuperación
debe restaurar los checkpoints del Run original; un envío nuevo no es equivalente. Comprueba la
previsualización de recuperación de LF antes de aplicarla. Algunas versiones de LF recuperan
estudios, pero reenvían los Works simples sin restaurar sus checkpoints: si la previsualización
no incluye un plan explícito de recuperación, solicita soporte nativo para ese caso en lugar de
copiar directorios internos o volver a calcularlo todo.

Los campos de espacio libre usan por defecto un límite de 18 millones de vóxeles, manteniendo
la separación de 1 Å; los dominios grandes necesitan bastante RAM (la receta de enriquecimiento
solicita 96 GiB). Un vóxel es una celda de la cuadrícula espacial. El límite rechaza una
cuadrícula demasiado grande; nunca la trunca ni reduce silenciosamente su resolución. Aumentarlo
no cambia los campos calculados. El enriquecimiento reconoce las claves históricas con límite
de dos millones y revalida esos mismos archivos comprobando sus parámetros físicos y los hashes
del archivo base. Sus metadatos conservan el límite original. Cambios físicos, como la separación
de la cuadrícula o el radio de la sonda, no permiten reutilizarlos.

Para zinc usa `experiments/preprocess/zinc/zinc_preprocess.yaml`. Encadena descubrimiento, selección,
preprocesado, validación y visualización. `zinc_discovery.yaml` permite adquirir y congelar evidencia
por separado. Los negativos de zinc exigen anotaciones negativas explícitas admitidas, nunca ausencia
de un átomo metálico. El descubrimiento puede obtenerlas automáticamente; se detiene **antes de
las descargas masivas de positivos** si no quedan al menos tres secuencias negativas distintas
(`minimum_negative_sequences: 3`). Varias estructuras de la misma secuencia no pueden ocupar
funciones independientes de entrenamiento, validación y test. Tres secuencias son solo un mínimo
necesario: la agrupación RAW posterior debe dejar al menos tres grupos negativos elegibles y cada
partición debe contener ambas clases. Los artefactos `negative-feasibility.json` y
`negative-discovery.json` explican los recuentos y exclusiones. Para obtener más negativos hace
falta evidencia experimental explícita adicional, no reetiquetar proteínas desconocidas.
Las etiquetas de ADN nunca entran en la tarea de zinc.

**Reproducir el benchmark de zinc del TFG.** Para comparar la representación y los modelos actuales
con el Trabajo de Fin de Grado (TFG), utiliza la receta histórica independiente:

```bash
lf clusters bootstrap citius-ctgpgpu16 --project .
lf validate experiments/preprocess/zinc/zinc_legacy_preprocess.yaml
lf run experiments/preprocess/zinc/zinc_legacy_preprocess.yaml --on citius-ctgpgpu16
# Tras publicarlo, validar en el mismo destino:
lf run experiments/preprocess/zinc/validate_zinc_legacy.yaml --on citius-ctgpgpu16
```

Publica **`wisdom-zinc-tfg@1`**, separado del descubrimiento moderno de zinc. Los seis originales
`data/zn_old/*.txt` no se modifican: `positives.txt` y `negatives.txt` aportan etiquetas;
`train.txt`, `val.txt` y `test.txt`, las particiones; `all.txt` comprueba la cobertura. Contienen
450 proteínas únicas: train 148 (74 por clase), validación 148 (74 por clase) y test 154
(77 por clase). No se ejecutan selección, balanceo, reagrupación por homología ni dilución.
**No** se aplica la vista train-25 de ADN. Particiones solapadas, etiquetas contradictorias,
IDs repetidos o estructuras ausentes detienen el proceso: nunca se quita o sustituye una proteína.

La receta utiliza los mismos parámetros base y atributos estructurales que la receta actual de
ADN `@9`: cinco escalas de curvatura, vecinos atómicos ampliados a 12 Å/K96, grafos de residuos,
profundidad/oclusión/espacio libre con máscaras, modos espectrales ampliados y una superficie
alternativa. Añade las representaciones seleccionadas de residuos ESM-2 y de átomos/residuos SLAE
del flujo enriquecido. El conjunto fijado `data/models` y
`data/external-models/slae/autoencoder.ckpt` deben estar en el destino, igual que para ADN.
La geometría utiliza procesos de CPU acotados; los encoders congelados solo usan la GPU lógica
concedida. El Zn nunca entra en las entradas neuronales. Son representaciones nuevas de la
población histórica, no una reproducción byte a byte de sus coordenadas o preprocesado originales.

Las entradas tipadas `legacy_train`, `legacy_validation`, `legacy_test`, `legacy_positives`,
`legacy_negatives` y la opcional `legacy_all` sustituyen a `design`; no se pueden combinar con
entradas de Selection moderno. El artefacto portátil `legacy-design` conserva los TXT originales,
sus hashes de contenido y todas las decisiones heredadas. Las primeras coordenadas observadas en
RCSB quedan en checkpoints duraderos reutilizables en recuperaciones nativas compatibles.
Se eligen de forma determinista el primer ensamblaje biológico declarado que contiene la cadena
histórica exacta y su primera copia, nunca por concordancia con la etiqueta o el sitio. Si falta
un ensamblaje, se informa del error: no se adivina otra cadena.
Las coordenadas almacenadas con ocupación cero se conservan para esta representación histórica y
su alineamiento geométrico al ensamblaje, pero no pasan a ser evidencia de donantes de Zn ocupados.
Una cadena completa con ocupación cero (por ejemplo `1C78_B`) queda advertida en el informe y sin
GT local utilizable. La selección moderna sigue exigiendo átomos ocupados para su alineamiento y
evidencia.

Antes del enriquecimiento estructural costoso, WISDOM comprueba la correspondencia entre cada
residuo observado y su posición depositada. Ambos lados usan el mismo diccionario químico de
Gemmi: residuos modificados como MSE, PTR, PCA y LLP se relacionan con sus aminoácidos de origen
conocidos M, Y, E y K. Los nombres originales de tres letras y la química atómica no cambian.
Los monómeros desconocidos siguen siendo X; una discrepancia real de identidad o posición sigue
produciendo un error que indica cadena, dirección del autor y posición depositada. No se utiliza
alineamiento aproximado de secuencias. La comprobación previa publica todos los IDs afectados.

Si el enriquecimiento se detiene después, la recuperación nativa compatible revalida y reutiliza
cada checkpoint de geometría, anotación y atributos terminado, incluidas las familias completadas
de proteínas fallidas. No reconstruye las superficies, curvaturas ni operadores espectrales de las
demás proteínas. Reutilizar no es instantáneo: la validación científica sigue comprobando la
exposición atómica al disolvente con las mismas muestras esféricas y reproduce el muestreo de las
vistas alternativas. Las muestras de exposición ya enterradas se descartan de las siguientes
pruebas de contención sin cambiar las áreas obtenidas. Las pruebas de enterramiento superficial
se ejecutan en lotes de pares cercanos; junto al umbral de contención conservan el cálculo
escalar original para mantener exactamente las decisiones de selección de puntos.
Los contadores nativos de restauración de
Map en el log describen este intento; el estado original de un registro guardado no demuestra que
se haya reconstruido. Un envío **nuevo** no es una
recuperación y puede empezar de cero. No borres el Job fallido ni ejecutes limpieza de almacenamiento
antes de recuperar. Si LF rechaza la recuperación porque renombró los TXT transferidos pese a
tener los mismos bytes, debe corregirse esa identidad en el framework sin alterar los archivos
científicos ni los manifiestos originales.

Las normales superficiales se comprueban con todas las esferas físicas cercanas, no solo con la
tabla limitada de átomos más próximos del modelo. Las vistas alternativas reutilizan sus puntos
reproducidos de forma determinista, antes del redondeo, para esta comprobación: guardar coordenadas
con 32 bits puede cambiar si un átomo entra en el corte estricto que usa la mezcla de normales.
La referencia debe coincidir con las posiciones almacenadas dentro de cuatro unidades de redondeo
float32, multiplicadas por el mayor entre 1 Å y la coordenada almacenada de mayor valor absoluto;
una unidad es la separación relativa `2^-23`. El umbral del coseno sigue siendo 0,99. Se mantienen las comprobaciones
de geometría almacenada y operadores, y el rechazo de normales invertidas. Esta corrección de la
validación no reescribe ningún archivo. Si una normal de la superficie base resulta dudosa, se
aplica la misma comprobación precisa solo si el muestreador original reproduce todos los puntos
almacenados exactamente tras redondear a float32.

Las etiquetas se marcan como `legacy_tfg_manifest`, **no como negativos confirmados
experimentalmente**. Los contactos recuperan referencias superficiales solo para evaluación, sin
cambiar etiquetas globales. Un positivo sin referencia local utilizable (GT), o un negativo
contradicho por un contacto con Zn, tiene GT local no disponible y una máscara de evaluación vacía:
no se inventan etiquetas locales. La validación comprueba miembros/etiquetas/particiones originales,
geometría, archivos de referencia y atributos activados. El informe declara
`homology_independence_verified: false`: las secuencias repetidas son avisos, no se reparan los
splits. PASS certifica este contrato de reproducción, no la curación moderna de negativos ni la
independencia entre homólogos. El flujo moderno conserva sus requisitos estrictos de evidencia.

La receta de zinc publica ahora **`wisdom-zinc@3`**. Discovery descarga depósitos mediante la
caché compartida de LF con hilos y después inspecciona los archivos con procesos acotados
(`structure_executor: process`; `workers` limita la concurrencia). Cada ensamblaje biológico construye un solo índice
espacial de donantes, una búsqueda de átomos proteicos cercanos, compartido por los resúmenes de
todas sus copias de cadena. Conserva cada candidato, alternativa, umbral y cadena sin contacto:
no elimina puentes RAW de fugas. Selection sigue registrando los contactos completos y utiliza
los mismos algoritmos independientes de tarea que ADN para similitud de secuencia/estructura,
particiones de grupos y diluciones anidadas. Las formas de sitios y evidencia negativa siguen
siendo específicas de cada tarea.

Las anotaciones descriptivas de UniProt solo se transfieren si la secuencia completa devuelta
coincide exactamente con la cadena depositada. Un resultado sin secuencia queda registrado en
`uniprot_sequence_unavailable` y no aporta funciones ni familias de UniProt. Esto no cambia la
etiqueta de zinc, que procede de la coordinación o de evidencia negativa explícita e independiente.
Los lectores paralelos de metadatos comparten **checkpoints duraderos de fuentes**, validados por
LF; Discovery copia cada respuesta distinta al registro portátil al terminar esa fase paralela.
Los inventarios públicos, respuestas de evidencia, metadatos y bytes de coordenadas descomprimidos
quedan fijados al adquirirse, no solo cuando termina todo el Work. LF gestiona bloqueos, descargas,
integridad y publicación atómica. La primera URL y su hash de bytes quedan fijos: si una fuente
perdida solo puede descargarse con otros bytes, la recuperación se detiene para no mezclar
revisiones públicas. Esto añade almacenamiento duradero junto a la caché de descarga descartable;
borrar los checkpoints de un Run fallido sigue destruyendo ese punto de recuperación. No implica
que envíos nuevos e independientes compartan datos.

`progress_log_seconds: 120` informa de que la fase sigue activa; LF muestra elementos
completados/totales. Descargas e inspecciones tienen registros nativos separados de recuperación
por depósito. Comprueba la previsualización nativa de recuperación de LF: debe conservar el
Execution/Run original y sus dependencias duraderas, incluso al aceptar expresamente una revisión
de código. `zinc_discovery.yaml` sigue siendo útil para congelar la adquisición por separado.
Un envío nuevo no hereda los registros del Run anterior. Los intentos antiguos que solo usaban
caché requieren migrar sus observaciones originales con verificación nativa antes de reutilizarlos;
WISDOM rechaza consultar un inventario nuevo sobre inspecciones anteriores. Los recuentos no
permiten recuperar bytes ya borrados. LF también necesita limpieza que respete dependencias y
esperas compartidas ante HTTP 429; WISDOM no implementa otro recolector de almacenamiento ni un
planificador de reintentos HTTP. Hasta disponer de esa corrección, no asumas que otro reintento
conservará las dependencias antiguas de caché. Tras completarlo, congela
las partes caras: pon `discover.skip: true` y descomenta `existing_raw` para reutilizar
`data/zinc/raw`; o elimina discovery, pon `select.skip: true`, descomenta `existing_design` y comenta
`raw_path` para reutilizar `data/zinc/design`. Mantén el preprocesado activo para publicar una
versión nueva con más recursos.

El preprocesado solicita **las mismas familias universales y parámetros científicos que ADN**:
contexto atómico, de secuencia y covalente, exposición al disolvente, candidatos 12 Å/K96, grafos
de contactos de residuos y carbonos alfa, propiedades de espacio libre, topología superficial,
extensión a 256 modos, superficie alternativa y cinco escalas de curvatura. Las recetas de ADN y
zinc seleccionan ahora solo contexto por residuo de **ESM-2**, la fuente conservada en V14/V15.
ESM-C, SaProt, las tablas de inicialización estática y el 3Di no utilizado están desactivados por
defecto, sin eliminar sus implementaciones. Prepara el mismo directorio fijado `data/models` en el
espejo remoto e instala o ejecuta bootstrap antes, igual que para ADN.
Si falta un recurso requerido, falla la comprobación inicial o la publicación en lugar de omitirlo.
Zinc también activa el checkpoint base revisado de SLAE en
`data/external-models/slae/autoencoder.ckpt`; copia los mismos pesos al espejo remoto antes del
envío y prepara con bootstrap el entorno aislado `slae`. Su extracción GPU procesa una proteína
cada vez después de generar los recursos estructurales en paralelo; `workers: 12` sigue aplicándose
a la geometría CPU, no a modelos SLAE concurrentes. La receta solicita una GPU explícitamente y
también usa su dispositivo CUDA lógico concedido para el extractor por residuo seleccionado.
El lector elimina Zn y otros átomos no proteicos **antes** de calcular geometría, exposición,
grafos, tokens y representaciones. Las fuentes exactas conservan Zn solo para evidencia y targets
de evaluación; esas coordenadas fuente y anotaciones no son entradas del modelo.

`python scripts/benchmark_zinc_discovery.py` reprodujo resúmenes idénticos y midió una inspección
unas 1,3/8,8/44 veces más rápida en ensamblajes sintéticos de 1/8/32 copias; las medidas están en
`benchmarks/zinc_discovery.json`. No son aceleraciones del pipeline completo. Con el límite
predeterminado de dos peticiones de coordenadas por segundo, solo espaciar 28.033 peticiones sin
caché requiere unas 3,9 horas; transferencia, metadatos, homología RAW completa y geometría final
añaden coste. No se muestrea la población, relaja el umbral de fugas ni aumenta la tasa de red.
Los recursos estructurales completos también cuestan más que la geometría sola: el YAML solicita
CPU/RAM/disco/GPU explícitamente y no usa dispositivos sin asignación.

La publicación predeterminada de Zn contiene vectores contextuales ESM-2 (480 valores por residuo
físico) y vectores SLAE (128 valores por átomo observado admitido y por residuo admitido).
Las máscaras identifican entidades ausentes o no admitidas; no se inventan átomos.
Los vectores por residuo pueden asociarse a sus
átomos mediante los índices de pertenencia almacenados, sin presentarlos como predicciones
independientes específicas de cada átomo. **No son embeddings de ESM-3**: que SLAE utilice el
paquete `esm` dentro del entorno aislado no significa que WISDOM extraiga otro modelo ESM-3.

Para repetir la comparación anterior, cambia `protein_model_kinds: [esm2]` por una lista que
incluya `esmc` o `saprot`; solo se comprueban, cargan y generan los modelos seleccionados. SaProt
solicita automáticamente su entrada 3Di oficial de Foldseek. Pon `protein_static_tables: true`
solo al estudiar inicialización con tablas estáticas. Estos valores son decisiones de las recetas:
el componente conserva sus valores históricos `protein_model_kinds: null` (los tres modelos) y
`protein_static_tables: true` para quienes ya usan el conjunto explícito. Las comprobaciones de
recursos requeridos siguen la selección, no presuponen todos los modelos. Aplica una receta
modificada a un `dataset_version` sin publicar; los datasets inmutables existentes conservan sus
recursos y tamaño. Cambiar estos valores no borra datasets antiguos, cachés ni pesos descargados.
En ADN, SLAE sigue siendo una acción separada en `enrich_atomic.yaml`.

El recurso `surface_environment` de V15 se genera para Zn con el mismo algoritmo proteico:
profundidad de recorrido (camino libre al exterior), profundidad euclídea (distancia directa al
exterior), oclusión cercana/lejana (fracción de rayos hacia fuera bloqueados antes de 4/15 Å) y
radio vacío (tamaño del espacio libre local asociado). Cada valor tiene su máscara de validez;
se mantienen explícitos cavidades cerradas, disolvente exterior y desplazamientos de asociación.
Todos los campos siguen el orden de puntos de la superficie base. Las alternativas `mask_only`
y `curvature_control` de V15 son controles durante entrenamiento sobre estos recursos y las
curvaturas existentes, no otro preprocesado ni etiquetas específicas de tarea.

## 4. Contratos técnicos

### 4.1. Arrays estructurales y anotaciones

Cada proteína tiene un **NPZ** universal: una colección comprimida de arrays numéricos con nombre.
No contiene objetos Python; se lee con `numpy.load(path, allow_pickle=False)`. Sus metadatos
identifican la fuente exacta, configuración y esquema. El `index.jsonl` del dataset vincula cada
miembro con su split, etiqueta global, archivo y datos específicos de tarea opcionales.

| Familia de arrays | Significado y uso posterior |
|---|---|
| `atom_positions`, `atomic_numbers`, descriptores químicos | Coordenadas atómicas en Å e identidades/propiedades fijas para el codificador atómico |
| `atom_edge_index`, distancias, enlaces y rangos espaciales | Grafo disperso covalente/espacial; cada par no dirigido se guarda una vez con `src < dst` |
| `surface_positions`, `surface_normals`, `surface_area_weights` | Frontera muestreada, direcciones unitarias hacia fuera y área representada en Å² |
| `surface_curvatures` | Curvatura media, curvatura gaussiana y curvedness por escala |
| `surface_atom_*` | Candidatos atómicos cercanos acotados, distancias/orientación y máscara de validez |
| `surface_neighbors`, máscaras y distancias | Vecindarios superficiales fijos para los componentes que los necesitan |
| `diffusion_*` | Masas superficiales, base espectral y derivadas dispersas utilizadas por DiffusionNet |

Las coordenadas se centran de forma coherente, no se reescalan arbitrariamente. Las curvaturas
describen cómo se dobla la superficie; sus transformaciones y el Shape Index calculado en ejecución
se implementan en el modelo superficial. Los huecos sin vecino se identifican con la máscara, no
con un átomo inventado. En la receta actual la tabla base histórica sigue en 6 Å/K32; la tabla
extendida independiente permite 12 Å/K96 sin reescribir ese contrato. Otras familias incluyen
contexto de secuencia/covalente, exposición atómica, grafos de residuos/superficie, extensiones
espectrales y muestreos superficiales alternativos.

Las representaciones opcionales de modelos de proteínas congelados son recursos declarados con
pesos y revisiones explícitos. El entrenamiento puede solicitarlos, pero su presencia no los activa
en silencio. Son distintos de la geometría genérica. Algunos extractores requieren pesos grandes
y bastante tiempo de CPU/GPU; activar todo tiene un coste.

El ground truth (GT) de ADN, la referencia local con la que juzgar una predicción, se guarda en un
archivo independiente ligado al orden exacto de puntos de la superficie base. El target hard es
binario; el soft varía de forma continua cerca del límite del contacto. Las zonas desconocidas no
se convierten en negativas. La geometría base no incluye átomos de ADN ni targets locales como
entradas neuronales. Zinc usa sus propias anotaciones sobre la misma geometría universal.

### 4.2. Modelos, supervisión y métricas

El flujo común es **grafo atómico → transferencia átomo-superficie → codificador superficial →
evidencia por punto → agregación de proteína**. El grafo intercambia información entre átomos
vecinos. La transferencia pondera los átomos cercanos para cada punto superficial. DiffusionNet
propaga información usando operadores geométricos precalculados. Una cabeza por punto produce
logits, puntuaciones reales que se convierten en probabilidades; el pooling las agrega en una
puntuación para toda la proteína.

`WisdomV1`, `WisdomV2` y `WisdomV3` exponen capacidades experimentales relacionadas, no la secuencia
V1–V13 de estudios científicos. Los codificadores de grafos, poolings y retornos alternativos son
parámetros explícitos. Los gates semánticos Hard-Concrete aprenden si una fuente de información
con significado es útil; su regularización desincentiva fuentes activas innecesarias. Esta
reorganización no cambia ecuaciones, atributos de módulos, nombres de parámetros en checkpoints,
valores predeterminados ni pesos de pérdidas.

Se entrena con etiquetas globales de proteína. Las pérdidas débiles opcionales restringen la
superficie usando esas mismas etiquetas y la geometría, no el GT local. Validación selecciona un
checkpoint según la política declarada; test solo se evalúa si está activado y nunca sirve para
ajustar. `atom_batch` conserva la pertenencia de átomos a proteínas al moverlos al dispositivo:
los codificadores direccionales y globales acotados no deben tratar el lote como una sola molécula.
Los límites por proteína no garantizan que toda la VRAM necesaria quepa.

Solo V9g permite explícitamente atención atómica global densa. Su receta admite 1 GiB por tensor
principal de códigos de rutas o puntuaciones de atención, frente al valor predeterminado de
128 MiB del constructor. En el dataset @6 registrado, el máximo es 942.127.232 bytes (10.852 átomos);
74 de sus 589 miembros superan el límite predeterminado. Las puntuaciones FP32 de dos cabezas y
los códigos int64 necesitan, cada uno, 8 bytes por par atómico. WISDOM indica pertenencia, número
de átomos, bytes necesarios y límite antes de reservar esos tensores. El control sin estructura
proporciona directamente códigos estructurales cero, sin búsquedas de rutas que no usa. No cambian
sus ecuaciones ni parámetros. La atención densa sigue creciendo cuadráticamente; capas, gradientes
guardados y Runs concurrentes requieren más memoria. Ni ese límite ni la precisión mixta garantizan
que quepa en una H100. LF trata actualmente este límite como parámetro científico: cambiarlo puede
impedir reutilizar Runs completados mediante recuperación nativa aunque las ecuaciones no cambien.
No se debe saltar ese control de identidad ni recortar proteínas en silencio.

Para `structural_global`, Training prepara los códigos exactos de rutas/grados de LF **antes de
la ejecución neuronal**, una vez por miembro y relación. LF gestiona bloqueos, integridad y
publicación atómica; la clave liga orden de nodos, topología, K activo e implementación, nunca
etiquetas ni GT. Capas, épocas y vistas de validación reutilizan esos archivos; los Attempts
compatibles pueden restaurarlos. La caché actual de LF incluye Run/semilla: no se comparte a
nivel de Study. La primera preparación puede tardar y muestra logs por miembro. Los códigos 0–31
se guardan y cargan sin pérdida como uint8, y se convierten a int64 una vez por lote en el dispositivo,
compartiéndose entre capas. Si faltan operadores preparados, se informa del error; no queda un
retorno silencioso a BFS lento dentro del `forward`. La revisión de checkpoints y extracción de
conceptos usan la misma preparación; no cambian los pesos ni los nombres del estado guardado.
No hace falta regenerar el dataset ni añadir parámetros YAML. La atención global sigue siendo
cuadrática: se elimina preparar el grafo repetidamente, no el coste intrínseco de atención en GPU.

Inspecciona métricas globales y locales. AUPRC resume precisión frente a sensibilidad al variar el
umbral y depende de la proporción de positivos. MCC mide acuerdo binario usando los cuatro
recuentos de la tabla de confusión. Las métricas superficiales comparan puntuaciones por punto con
GT disponible; ponderar por área evita que un muestreo denso domine. Los valores indefinidos
conservan sus recuentos de soporte en lugar de ceros inventados. Revisa el objetivo de cada estudio:
una métrica diagnóstica superficial no es automáticamente un objetivo de optimización. La memoria
explica las puntuaciones locales normalizadas, utilidades combinadas y sus límites.

### 4.3. Controles radiales y de anchura de SchNet

V9d deja abierto si SchNet mejora por su representación de la distancia o por su capacidad.
El nuevo estudio **V9f SchNet** compara distancia directa y características gaussianas, anchuras
atómicas internas 64/128 y metadatos de arista completos/restringidos. V9g pasa a nombrar la atención
dispersa/global; las ejecuciones históricas `wisdom-v9f` siguen siendo evidencia de atención.
El nuevo Work se llama `wisdom-v9f-schnet`, deliberadamente distinto. El YAML histórico sin cambios
y su correspondencia SHA-256 se conservan en `experiments/v9/campaign_migration.json`; no se renombran
resultados guardados.

Una función de base radial gaussiana (RBF) responde suavemente alrededor de una distancia elegida.
Para una arista de longitud $d$ en angstroms, R características sitúan sus centros $\mu_k$ entre
0 y 6 Å. Con separación $\Delta=6/(R-1)$ Å, sus valores son:

$$
b_k(d)=\exp[-\gamma(d-\mu_k)^2],\qquad \gamma=0.5/\Delta^2.
$$

En RBF-16, la separación es 0,4 Å y gamma es 3,125 Å⁻², exactamente la base de LF usada en V9d.
Para $d=2$ Å, los centros 1,6, 2,0 y 2,4 Å responden aproximadamente 0,607, 1 y 0,607. El modo
directo aporta el único valor $d/6=1/3$. Ninguno añade información geométrica nueva: el experimento
comprueba si la expansión facilita aprender filtros, no supone que más características sean mejores.

Para cada relación, la red de filtros de LF recibe características radiales `[E,R]` y los atributos
de arista existentes con gates `[E,A]`, y genera filtros `[E,H]`. E cuenta aristas dirigidas, A vale
3 en la relación covalente y 4 en la espacial, y H es la **anchura atómica interna**. Los estados
proyectados del átomo emisor `[E,H]` se multiplican canal a canal por esos filtros; los mensajes se
suman en el receptor antes de la actualización residual de LF. Ambos modos usan el mismo filtro
de dos capas con SiLU. Su número de parámetros es `H*(R+A+1) + H*H + H`: cambiar R también cambia
la capacidad. Se mantiene el atributo escalar de distancia original, incluso en el modo directo;
no se elimina ninguna otra fuente.

`atomic_radial_encoding` usa `gaussian` por defecto; `direct` aporta un canal radial.
`atomic_radial_count=16` es el valor gaussiano predeterminado; V9f prueba 8/16/32.
`atomic_hidden_dim=null` hereda `hidden_dim`; SchNet admite 64/96/128. `atomic_output_dim=null`
también hereda `hidden_dim` y debe coincidir con él: los estados reducidos usan una proyección
aprendible 64→128; si las anchuras coinciden no se añade una capa. Las representaciones químicas,
la transferencia, DiffusionNet y las cabezas mantienen sus anchuras originales. El corte coseno
espacial sigue en 6 Å; las aristas covalentes no añaden un corte, ni fuera del intervalo de centros.
Con gate de distancia g, la entrada radial es `g*b(d)` y el soporte `1+g*(cutoff(d)-1)`; g=0 elimina
la información de distancia, no la relación ni sus otros atributos. `schnet_radial` conserva
distancias y química, pero hace indisponibles los mismos cinco metadatos que en V9d.

LF registra curvas de calidad/coste y cobertura de semillas emparejadas.
`atomic-encoder-audit/initialization.json` guarda centros, gamma, anchuras, recuentos de parámetros
y un hash de los pesos iniciales comunes. También cuenta multiplicaciones-acumulaciones de las
capas lineales del filtro por arista y de la
proyección de salida por átomo; este recuento parcial no representa FLOPs del modelo completo ni tiempo.
Tras restaurar el mejor checkpoint **seleccionado por G**, `radial-filters.json`, `.npz` y `.png`
inspeccionan 121 distancias en [0,6] Å, separando relaciones,
capas y perfiles de metadatos declarados. Conservan respuestas antes/después del soporte y gates
de relación, amplitudes, sensibilidad por diferencias finitas y correlación entre canales. Estos
filtros latentes no son potenciales energéticos físicos. El GT local sigue siendo solo de evaluación.
Los picos CUDA asignado/reservado miden el Run completo, no solo el encoder; la mediana de época
completa excluye la primera e incluye la validación programada. El benchmark sintético independiente
`scripts/benchmark_schnet.py` mide por separado forward y backward del encoder.
La desviación típica y los cuartiles de AUPRC entre proteínas positivas muestran su dispersión;
la desviación no está disponible con menos de dos proteínas evaluables. No miden incertidumbre
entre semillas ni cambian la media S.

El [catálogo de campañas](experiments/README.md) describe las 13 celdas y los comandos por bloque/celda.
El script opcional `scripts/analyze_schnet_campaign.py` lee metadatos nativos del checkpoint y calcula
los contrastes emparejados predefinidos respecto a la celda 04, anchura, metadatos e interacciones,
con gráficas y tablas de coste relativo. Conserva los soportes ausentes y no sustituye las decisiones
ni la exploración de parámetros de LF.
Cuatro semillas emparejadas no establecen conclusiones universales; no infieras interacciones completas
metadatos×RBF a partir de celdas no medidas, ni ahorro de memoria a partir de parámetros solamente.
La matriz principal de 13 celdas no incluye un control extra de anchura de filtros con capacidad igualada.

V10 fija ahora **SchNet con metadatos completos, RBF-16 gaussianas, 64 canales internos y 128 de
salida**, a partir de la celda 08 de desarrollo de V9f (`execution-d384d8f6b5064a26`). Su S con
cuatro semillas es 0,2932 frente a 0,2966 de directa/restringida/128, con 87.488 frente a 281.728
parámetros entrenables del encoder, menores picos CUDA observados y menos activación superficial
falsa en proteínas negativas. Es un compromiso provisional de calidad/capacidad, no un ganador
confirmado ni una aceleración demostrada: G es 0,7030 frente a 0,7134 del baseline contemporáneo
original, y los tiempos de época en hardware compartido no favorecen de forma consistente al modelo
estrecho. Todos los controles e intervenciones de V10 entrenan esta misma arquitectura desde cero;
el mapa de entradas vacío significa SchNet sin fuentes adicionales, no el encoder baseline anterior.
V10 conserva sus semillas y la entrada enriquecida `wisdom-dna-reduced@8`; los resultados con @6
no certifican equivalencia con @8. Los parámetros, hashes de evidencia, diferencias emparejadas y
limitaciones se guardan en [el registro de selección de V10](experiments/v10/backbone_selection.json).

[V10e](experiments/v10/wisdom_v10e.yaml) declara ahora 16 configuraciones de fuentes y cuatro
semillas pareadas de desarrollo, 64 ejecuciones. Repite el control sin fuentes nuevas y las
referencias aisladas bajo una identidad común de código y datos, compara nombre con nombre y
pareja residuo–átomo como alternativas separadas y prueba interacciones A/B/C,
combinaciones seleccionadas con SASA y adiciones anidadas de identidad exacta y motivos. Todos los
gates de fuente siguen siendo aprendibles con el denominador de referencia fijo. Un efecto pequeño
o incierto en ADN no elimina una fuente del modelo reutilizable, y un gate abierto no acredita
utilidad en otra tarea. La selección sigue siendo de desarrollo, sin evaluar test ni abrir otro
barrido de arquitectura u optimización.

La política de fuentes/gates posterior a V10f sigue registrada en
[source_gate_selection.json](experiments/v10/source_gate_selection.json). La continuación actual
tras V10g/V10h queda en [representation_selection.json](experiments/v10/representation_selection.json):
`embedding_initialization: current`, SchNet con metadatos completos y 64 canales internos/128 de
salida, y `atomic_representation: {sources: [atom_name, residue_atom], identity_dim: 8,
force_sources: learned, external_model: esm2, external_mode: context, context_dim: 32}`.
ESM-2 aporta contexto congelado por residuo; en este modo **no inicializa** la tabla de residuos.
La tabla habitual sigue aprendiendo. No se combina una tabla de SaProt con contexto de ESM-2.
Los gates conservan coeficiente 0,04, denominador 41, actividad inicial 0,95 y ninguna programación.
Es una decisión de desarrollo, no superioridad universal ni descarte automático de fuentes.

[V10i](experiments/v10/wisdom_v10i.yaml) vuelve a comparar tres poolings con esa representación fija:
MAX, LSE fijo ponderado por área (`beta=0.25`) y atención simple ponderada por área (anchura 256).
Cuatro semillas nuevas pareadas, 307/311/313/317, producen **12 entrenamientos obligatorios**.
Todas las familias conservan cabeza única, calor de 6 Å y prior regional 0,3. No se añade HPO ni
se importan pérdidas V6 distintas para cada familia. El YAML de ejemplo reproduce el ejecutable.
Se inspeccionan S, G y evidencia negativa por separado, además de pico de memoria asignada,
segundos por época completa y épocas realizadas. MAX sigue **provisional** en V11/V13 hasta que
V10i aporte evidencia para sustituirlo.

Los diseños de entrenamiento V11/V13 heredan contexto, inicialización y SchNet seleccionados,
sin volver silenciosamente al encoder relacional anterior. Conservan sus factores de
transferencia/superficie, datos, semillas exploratorias y presupuestos. Los cierres usan
401/409/419/421 en V11f y 503/509/521/523 en V13f, evitando las semillas de selección de V10i.
V12 solo hace inferencia. Su receta selecciona exactamente el contexto ESM-2 de la ejecución
V10h completa `execution-6c675d5c9f10d971`, con sus cuatro semillas, y restaura pesos/gates guardados.
Antes de esa revisión se registra el estudio portable con `lf import`; los protocolos aislados
reconstruyen superficies compatibles desde el archivo base exacto. No se introducen parámetros
de entrenamiento. Los YAML/resultados
históricos de V10g/V10h permanecen intactos; editar los archivos no ejecuta ni envía trabajos.

La inicialización de entradas externas escribe un único directorio `atomic-input-audit`.
Su archivo `normalization.json` recoge estadísticas ajustadas solo con las proteínas de
entrenamiento seleccionadas; `static-table.json`, cuando corresponde, registra cómo la tabla
preentrenada inicializa la representación aprendible de los residuos. Ambos pueden coexistir.
En modo `replace`, las características contextuales sustituyen esa representación: se conserva
su entrada de anchura cero en el checkpoint para poder cargarlo, pero no se ejecutan búsquedas
ni retropropagación sobre esa tabla vacía.

### 4.4. Selección de fuentes por grupos

[V10f](experiments/v10/wisdom_v10f.yaml) estudia si los gates Hard-Concrete pueden seleccionar
fuentes útiles manteniendo disponibles las siete entradas químicas añadidas. Un gate es un
escalar aprendido y compartido entre proteínas, no una decisión distinta por átomo. Su probabilidad
analítica de actividad indica la probabilidad de una muestra de entrenamiento distinta de cero;
su apertura determinista en inferencia es otra magnitud. Por ejemplo, una actividad inicial de
0,50 produce una apertura de aproximadamente 0,102, no de 0,50. Cero exacto significa cierre;
un valor positivo menor significa atenuación, no eliminación.

El modelo de cinco escalas tiene 41 gates históricos y siete químicos añadidos. Sus costes directos
se reparten en tres grupos disjuntos: S contiene 20 gates de curvatura, C los siete químicos y O
los otros 21 históricos. `surface.atom_context` pertenece a O porque controla contexto atómico,
no la forma superficial. La actividad efectiva de un hijo sigue siendo su probabilidad multiplicada
por las de sus ancestros. Los padres también tienen su propio coste directo; cada gate se cuenta
una sola vez, sin añadir gates padres para los grupos.

Denotando por lambda los coeficientes de grupo y por p las probabilidades efectivas de cada ruta,
la penalización adicional de entrenamiento es:

$$
L_{gate} = \frac{\lambda_O\sum_{i\in O}p_i
                     +\lambda_S\sum_{i\in S}p_i
                     +\lambda_C\sum_{i\in C}p_i}{41}.
$$

Aquí i identifica un gate y el denominador permanece en 41 aunque se añadan u omitan entradas
químicas. Este término ya ponderado se suma una vez a la pérdida de proteína y a las restricciones
débiles basadas en etiquetas de proteína; **no vuelve a multiplicarse por `gate_lambda`**. El GT
superficial nunca entra en la pérdida ni en la selección del checkpoint. Con todos los coeficientes
a 0,04 se reproducen la penalización y los gradientes de referencia de V10e. AdamW no aplica
weight decay, su penalización de magnitud de pesos, a los parámetros de los gates.

Los argumentos escalares opcionales de Training `gate_lambda_surface`, `gate_lambda_chemical` y
`gate_lambda_other` activan esta política. Los coeficientes de grupo omitidos pasan entonces a
0,04. `gate_initial_active_surface` y `gate_initial_active_chemical` valen por defecto 0,95; O
conserva `gate_initial_active`. Sin argumentos de grupo, la regularización histórica no cambia.
`atomic_sources` admite nombres canónicos separados por comas y `atomic_force_sources` admite
`learned`, `open` o `closed` solo para las fuentes añadidas. Estas coordenadas escalares permiten
enumerar perfiles completos con `sweep.candidates` nativo de LF, sin un selector de perfiles oculto.
También se admite el mapping equivalente `gate_policy`, pero no se puede mezclar con controles
escalares de política. El YAML documenta los defaults completos y el diseño exacto de 20 perfiles.

`gate_group_warmup_fraction` y `gate_group_ramp_fraction` usan el progreso nominal
`(epoch-1)/epochs`. Durante el warm-up solo S/C se fuerzan abiertos y sin coste directo; O sigue
aprendiendo y penalizado. Durante la rampa, S/C vuelven al muestreo estocástico y sus coeficientes
crecen linealmente. Estas opciones no se pueden combinar con el calendario global antiguo.
El checkpoint seleccionado guarda y restaura los overrides exactos, incluso un warm-up incompleto:
no debe presentarse como cierre aprendido. Cerrar entradas escribe ceros tanto en valores como
en máscaras de validez; no reduce físicamente la red ni garantiza entrenamientos más rápidos.

Cada Run habilitado publica un artefacto `gate-audit`: `manifest.json` identifica inventario,
anchuras, probabilidades iniciales, política, identidad del dataset, semillas y hashes del código;
`epochs.jsonl` registra actividad por gate, recortes reales del muestreo y gradientes anteriores
al paso del optimizador una vez por época. `initial.json`, `final.json` y `selected.json` separan
inicialización, último estado entrenado y checkpoint elegido por G de validación de proteína.
La procedencia nativa del attempt de LF aporta la identidad del código/paquete. Git puede no estar
disponible en el worker empaquetado; el HEAD y estado de cambios de preparación se registran en
`experiments/v10/gate_study.json`, sin presentarlos como un commit limpio exacto de ejecución.
Los hashes de código identifican los archivos inspeccionados. Los costes y cierres de grupo también
se registran como métricas de LF, separados de la calidad predictiva. Localizar bien con gates
abiertos no demuestra selección; cerrarlos destruyendo la localización tampoco es un éxito.
El benchmark de ADN por sí solo no demuestra utilidad universal.

### 4.5. Auditorías geométricas congeladas y entradas multiescala

V12–V15 separan estabilidad numérica, organización molecular y nueva información física de la
superficie. Ninguna cambia etiquetas, particiones ni la supervisión débil. V14/V15 mantienen
salida SchNet y anchura superficial de 128, una pasada superficial, MAX, Heat6 y existencia
regional 0,3 salvo que el YAML declare expresamente otro contraste. La validación por proteína
sigue eligiendo la época; el GT local no aporta gradientes. Son comparaciones implementadas,
no evidencia de que añadir información mejore la predicción.

V14/V15 utilizan la corrección de valores de rango 16 de V11b como continuidad provisional
de desarrollo, `transfer_options: {message: low_rank, rank: 16}`. Conservan DiffusionNet de una
lectura con descriptor de gradiente, 32 modos, soporte corto de 16 átomos/4 Å y ninguna rama
de mensajes locales. Se selecciona un cambio, sin acumular candidatos V11/V13 no contrastados
juntos ni declarar un modelo final confirmado. Los barridos de entorno repiten el diccionario
superficial completo, porque LambdaForge sustituye esos valores sin fusionarlos implícitamente.
Los selectores `@9` y SLAE opcional deben existir en el destino antes de ejecutar.

Desde V14b, los YAML V14b–f y V15 adoptan `context_dim: 64` para ESM-2 y
`atomic_hidden_dim: 128`, conservando salida atómica y anchura superficial de 128. Es la
opción de desarrollo 64/128 de V14a con el presupuesto H200 disponible, sin superioridad
confirmada frente a la alternativa compacta 16/64. V14a/a2 conservan sus diseños ejecutados.
Los diccionarios completos del barrido repiten el nuevo ancho contextual; el control solo SLAE
sigue sin ESM-2, la proyección SLAE permanece en 32 y el grafo residual de V14, cuando
se activa, conserva su propio ancho interno de 64. No cambian semillas, retiradas de
fuentes, contrastes de comunicación ni controles geométricos.

V14c seleccionó provisionalmente los contactos Cα (`residue_graph: {mode: contact}`)
para V14d–f y los primeros borradores de V15. V14d los fija en trece retiradas de fuentes
y añade un control sin rama (15 configuraciones × cuatro semillas = 60 entrenos).
Retirar `external_context` anula el contenido ESM-2 antes de las rutas directa y de
mensajes entre residuos; cerrar solo su gate no retiraría ambas. V14e compara seis
recorridos de comunicación y control. V14f repite sin rama, por nodo, contactos y mensajes
vecinales nulos con cuatro semillas nuevas (16 entrenos). Contactos reducen S superficial
operativa frente a la ausencia de rama en las cuatro semillas, mientras G global casi no
cambia. Por ello los cuatro YAML V15 omiten ahora `residue_graph` en su referencia molecular
común y solo varían la geometría superficial declarada. Conservan ESM-2 64, SchNet 128,
transferencia de rango 16, MAX, Heat6 y existencia regional 0,3. El retorno sin resumen
atómico de V14e y la combinación con SLAE siguen sin confirmar y no se acumulan en V15.
Los diseños y resultados V14a–f ejecutados permanecen intactos.

Cada experimento V12–V15 usa cuatro semillas fijas de modelo, compartidas por todas sus
alternativas. V12 selecciona cuatro checkpoints guardados, sin entrenar; sus semillas de vista
controlan el muestreo geométrico, no las réplicas del modelo. Los cierres mantienen sus semillas
nuevas e independientes. V14a ejecuta ahora las doce combinaciones de anchura con cuatro
semillas (48 entrenos), en lugar del cribado inicial de dos semillas propuesto en la memoria.

| Fase | Entrada o intervención | Qué aísla la comparación |
|---|---|---|
| V12 | Checkpoint congelado; identidad, tensores rígidos, reconstrucción girada, fase, origen de rejilla y resolución | Invariancia del modelo frente a dependencia de la discretización |
| V13 | Espectro, lecturas difusivas simultáneas, mensajes locales dispersos y encoders existentes | Procesamiento superficial, sin cambiar información atómica |
| V14a/a2 | Proyección ESM-2 16/32/64/128 × anchura atómica 32/64/128; controles alineado/media/permutado aparte | Compresión, capacidad y correspondencia local de residuos |
| V14b | SLAE atómico congelado opcional con/sin ESM-2 y un control sin mensajes atómicos adicionales | Representación atómica frente al contexto por residuo actual |
| V14c/d/e/f | Grafo de residuos, controles por nodo/nulos, retirada de fuentes, comunicación acotada y confirmación | Comunicación física frente a parámetros o rutas de información compensatorias |
| V15a/b/c/e | Profundidad, oclusión a dos horizontes, radio vacío asociado; controles de máscara/capacidad | Descriptores físicos distintos, seguidos de una combinación provisional |

**Preparar los datos.** La receta de ADN apunta ahora a `wisdom-dna-reduced@9`, conservando
train-25, validación completa y la exclusión actual de test. Añade assets a la misma representación
física; no sobrescribe @8. Como alternativa, `experiments/preprocess/enrich_structures.yaml`
enriquece @8 sin recalcular superficies base, anotaciones ni embeddings externos por residuo.
Elige una sola vía de publicación para @9. Los experimentos históricos mantienen sus selectores.
Los YAML de entrenamiento @9/@SLAE no pueden resolver el destino hasta que se publique.

El asset `residue_graph` conserva contactos entre átomos pesados y añade por separado contactos
Cα hasta 12 Å con 32 rangos de vecinos dirigidos. Cα es el carbono alfa del esqueleto, una
referencia física posible de cada residuo; no el átomo más próximo de su cadena lateral. V14 usa
hasta 16 contactos dentro de 10 Å y conserva relaciones peptídicas/covalentes verificadas fuera
del radio. La unión simétrica puede producir más de 16 vecinos entrantes. Cada pareja no dirigida
se almacena una vez. La continuidad peptídica procede de enlaces C–N reales, no de números
consecutivos. Las posiciones Cα ausentes y marcos N–Cα–C degenerados mantienen máscara: no se
sustituyen por el origen ni una orientación inventada. Los índices identifican residuos físicos
de copias concretas de cadenas.

El modelo de residuos usa capas SchNet de LF con pesos propios, normalmente dos capas y anchura
64, y adapta la salida a la anchura contextual. Conserva la ruta contextual original: una última
corrección inicializada a cero con multiplicador uno recupera la referencia y puede aprender
desde el primer paso. El control por nodo no consulta otros residuos; el control nulo conserva
el grafo y capacidad de filtrado, pero suprime mensajes. La comunicación descendente refina el
contexto antes de la GNN atómica; el retorno añade una media atómica por residuo y un adaptador,
no otra GNN oculta; la rama paralela envía un mensaje residual ponderado por distancia hacia la
superficie antes de la misma proyección y única DiffusionNet. Sus candidatos proceden de todos
los átomos pesados próximos, deduplicados por residuo, no de la lista atómica activa truncada.
`drop_sources` retira entradas conservando anchuras nominales; no compacta parámetros. Retirar
ESM-2 también anula su entrada al grafo de residuos. Retirar una entrada no borra información
que el extractor preentrenado ya haya consumido.

**Nuevos campos físicos.** `surface_environment` se vincula al hash del padre y conserva el
orden exacto de puntos. Guarda distancias no negativas en Å, fracciones de oclusión en [0,1],
máscaras, indicadores de cavidad cerrada/solvente exterior, offsets efectivos y distancias de
asociación a centros. Solo átomos proteicos y radios van der Waals definen obstáculos; ninguna
coordenada de ligando, máscara funcional o etiqueta elige un bolsillo. Un valor no disponible
es cero únicamente junto a una máscara falsa.

La profundidad sigue segmentos libres en un grafo de vóxeles de 26 vecinos hasta celdas libres
fuera de una aproximación por planos de soporte de la envolvente convexa de esferas expandidas.
No es distancia al centro de la proteína ni una recta que atraviese sus obstáculos. Las cavidades
inaccesibles no tienen profundidad exterior finita. Vóxel de 1 Å, 96 direcciones de soporte y
dos millones de vóxeles por proteína son valores explícitos; superar el límite causa error, no
un cambio silencioso de resolución. El stencil finito es anisótropo y la conectividad de cuellos
estrechos depende de la resolución. Se conserva la distancia euclídea al mismo exterior discreto
como control, no como sustituto.

La oclusión cuenta rayos del hemisferio exterior que chocan con esferas expandidas antes de 4 y
15 Å, con 64 direcciones deterministas. No se duplica su complemento como otra fuente. La
cuadratura angular finita usa un marco declarado ligado a ejes globales y no es exactamente
invariante a giros. El radio vacío asociado es la mayor holgura de un máximo interior de rejilla
visible y compatible con la pared dentro de 8 Å. No es volumen de cavidad ni holgura medida
sobre la SAS, que sería casi el radio fijo de sonda. Se excluye el solvente exterior no acotado;
las asociaciones inexistentes mantienen máscara. V15 compara valores reales con máscara sola
y controles de curvatura de igual anchura; normaliza solo valores válidos de train seleccionado.
Son aproximaciones acotadas que necesitan los pilotos geométricos de la memoria, no detectores
funcionales de bolsillos validados. No se sustituyen silenciosamente por extensiones opcionales
de cuellos, sondas múltiples o solvers adaptativos.

**Auditar predicciones fijas.** `experiments/v12/wisdom_v12.yaml` usa checkpoints importados
nativamente, sin optimizador. `view_audit.protocols` declara perturbaciones aisladas, `view_seeds`
sus semillas reproducibles y `maximum_proteins` el número auditado (cero significa todas). La
fase usa un desplazamiento SHA-256 por átomo físico; la rejilla cambia solo su origen relativo.
Las razones 0,8/1,25 de resolución y la reconstrucción girada regeneran todos los operadores
dependientes de puntos mediante archivos gestionados nativamente. Ningún punto nuevo hereda
una etiqueta por índice. La auditoría inicial necesita el contrato original de entradas por punto;
tablas extendidas, espectros adicionales, grafos locales y campos ambientales nuevos deben
regenerarse coherentemente antes de auditar esos modelos. Las combinaciones no soportadas
fallan en lugar de aparentar estabilidad usando arrays obsoletos.

En ambas direcciones se informa área comparable, errores de logits/probabilidades, correlación
de rangos e intersección sobre unión del 5/10 % superior por área, antes y después de Heat6.
Las correlaciones de mapas constantes permanecen no disponibles. Campos constantes, suaves y
localizados ligados a un átomo físico calibran el error de interpolación. A1 compara las mismas
filas físicas; las reconstrucciones interpolan por distancia inversa al cuadrado entre vecinos
con normales compatibles. Estabilidad no equivale a acierto: un mapa constante erróneo puede
ser estable. G/S de la partición original se evalúan normalmente; la auditoría no inventa GT
local para la nueva vista interpolando la anotación antigua.

**Preentrenamiento atómico opcional.** El checkpoint base revisado de SLAE está en
`data/external-models/slae/autoencoder.ckpt`, con URL, revisión, tamaño y SHA-256 en
`checkpoint.json`. Es el modelo base, no el ajustado con PDB. SLAE usa atom37, un vocabulario
fijo de nombres atómicos por residuo físico. El adaptador selecciona solo átomos pesados estándar
observados, verifica parejas únicas residuo/nombre, ordena expresamente y devuelve estados
escalares de 128 canales al orden original. Átomos/residuos no soportados permanecen en WISDOM
con máscaras; no se reconstruyen ni se borran silenciosamente del benchmark.
La cobertura se registra por cadena física, nombre atómico y nombre de residuo. Los miembros
sin soporte conservan máscaras totalmente falsas; una correspondencia física ambigua provoca
un error en lugar de inventar la asociación.

Las dependencias compiladas de PyG, la restricción `numpy<2` y el paquete `esm` de SLAE difieren
del entorno normal. `pyproject.toml` declara ahora el entorno independiente
`tool.lambdaforge.tool_runtimes.slae`: Python 3.11, Torch 2.6.0/CUDA 12.4,
bibliotecas PyG precompiladas compatibles, ESM 3.2.1 y Transformers 4.48.1. WISDOM conserva
su dependencia de Transformers 5; no instales SLAE en ese entorno con pip ni con el antiguo
extra `atomic-representations`. LambdaForge debe ofrecer `self.tools.runtime`; utiliza la versión
del repositorio que implementa herramientas aisladas, no una instalación anterior con el mismo
número de versión.

Desde la raíz del repositorio, prepara el destino una vez, valida el Work y lánzalo tú mismo:

```bash
lf clusters bootstrap citius-ctgpgpu16 --project . --dry-run
lf clusters bootstrap citius-ctgpgpu16 --project .
lf validate experiments/preprocess/enrich_atomic.yaml
lf run experiments/preprocess/enrich_atomic.yaml --on citius-ctgpgpu16
```

El primer bootstrap descarga e instala los paquetes aislados; los trabajos posteriores compatibles
los reutilizan. No extrae proteínas ni carga pesos. Se preparan todas las herramientas declaradas,
aunque el siguiente Work no utilice SLAE. Conserva el checkpoint revisado en
`data/external-models/slae/autoencoder.ckpt` localmente y en la ruta equivalente del proyecto
replicado en el destino. La entrada de fichero tipada verifica sus bytes; un checkpoint no equivale
a tener instalado el extractor. El dataset padre `wisdom-dna-reduced@9` debe estar registrado y
disponible en el destino.

La carga de pesos construye solo el encoder público de SLAE, con su configuración incluida.
El checkpoint base verificado contiene dos campos antiguos que no son pesos aprendidos:
`allegro.env_sum_normalizations` y `edge_eng_sum.per_edge_scales`. La implementación fijada
recalcula la normalización por vecinos en cada pasada y ya no almacena esos campos. WISDOM
excluye únicamente esos dos nombres y carga todos los demás campos del encoder con `strict=True`:
los pesos ausentes, desconocidos o de dimensiones incorrectas siguen provocando un error.
No modifica el checkpoint ni construye el decodificador de coordenadas, que no se utiliza.
Así conserva la regla de inferencia del encoder fijado, sin aceptar silenciosamente un modelo
parcial cualquiera.

El Work comprueba el entorno, la GPU lógica y el checkpoint antes de procesar proteínas; después
ejecuta el fichero independiente `SLAEExtraction.py` mediante `self.tools.runtime("slae")`. Cada
invocación carga los pesos congelados e intercambia ficheros NPZ explícitos, sin importar WISDOM
en el entorno aislado. El proceso principal valida valores finitos, máscaras y correspondencia
física de átomos antes de aceptar el checkpoint nativo del fichero. Los ficheros correctos quedan
disponibles para una recuperación nativa compatible; el Work no instala paquetes ni reentrena
ocultamente. El paso publica `wisdom-dna-reduced-slae@1` conservando las proteínas, particiones,
etiquetas y archivos existentes de @9; no repite la geometría ni Selection. La publicación nativa
puede copiar los archivos originales: reserva espacio para otra ubicación completa, no solo para
las pequeñas representaciones añadidas.

La receta incluida exige CUDA y respeta `CUDA_VISIBLE_DEVICES`. Usar CPU requiere revisar
explícitamente las versiones de Torch/vision/PyG y sus fuentes, configurar `torch_cuda="cpu"`,
repetir el bootstrap y seleccionar `atomic_device: cpu` y `resources.gpu: 0`; cambiar solo el
parámetro del Work no basta. El cargador corregido se comprobó en CPU en el destino con el
checkpoint base real: un grafo sintético de ocho átomos produjo arrays finitos [8,128] de átomos
y [2,128] de residuos, exactamente iguales a los obtenidos con el cargador oficial. Los datos de
la comprobación están en `benchmarks/slae_encoder_loading.json`. Esto comprueba la carga y una
pasada pequeña, no la cobertura de proteínas reales, el rendimiento CUDA ni la calidad preentrenada.
Aún hacen falta una extracción completa en el destino y un piloto de cobertura del benchmark;
la validación del esquema, la simulación y las pruebas con sustitutos no demuestran esos resultados.
El solapamiento del preentrenamiento con familias del benchmark sigue desconocido.
ESM-AA no es un alias de ESM-2: su ejemplo público de contactos
no establece correspondencia de tokens con átomos observados, por lo que no se habilita un
extractor ficticio listo para ejecutar.

### 4.6. Consolidación y supervisión de regiones negativas

Las recetas ejecutables de V16 están en `experiments/v16`. V16a compara la referencia histórica de
contexto ESM-2 64 / estado SchNet 128 con la ruta compacta 16/64 en cuatro semillas pareadas.
La ruta compacta es el punto de partida para la consolidación por su calidad media comparable y
menor consumo de memoria. V16a **no** añade automáticamente SLAE, retorno entre residuos o
descriptores de V15. La receta actual de V16b retira ESM-2 (`external_model: null`,
`external_mode: none`, `external_context_dim: null`), pero mantiene la anchura compacta de SchNet
64, las tablas aprendidas de identidad de átomo/residuo y el resto de la red. Explora profundidades
atómica, de proyección y superficial 1–6, dropout, tasa de aprendizaje 0.0003–0.003 y decaimiento
de pesos. El HPO adaptativo nativo limita la búsqueda a 128 candidatos, 300 Runs totales y 168 horas.
Las cuatro semillas de desarrollo forman una reserva de réplicas: se exige una al principio y se
asignan las demás de forma adaptativa; tres finalistas reciben tres semillas de confirmación
distintas. La poda por curvas y la paciencia habitual por G siguen activas. Son decisiones de
presupuesto y diseño, no evidencia de que retirar ESM mejore la localización. El estudio con ESM
que ya está en ejecución no cambia: inicia esta revisión como **estudio nuevo**, nunca recuperes
ni amplíes el anterior con la nueva representación. Sirve el mismo dataset @9; no se consumen sus
archivos ESM sin usar. `external_context_dim` normalmente admite un valor positivo solo si la
proyección externa está activa; debe ser null cuando se desactiva.

V16a2 comprueba por separado dos señales abiertas de V15 con cuatro semillas nuevas respecto a
esa campaña. Compara la referencia 64/128 con las máscaras de oclusión solas y con una rama que
vuelve a leer dos canales de curvatura ya transformados bajo esas máscaras. Así contrasta ruta y
capacidad, sin añadir un descriptor físico: V15a y V15c no reprodujeron la mejora local de V15b.
Un segundo par compara la máscara de validez de profundidad con el valor real; V15e mejoró la
localización media con profundidad, mientras V15a perdió en las cuatro semillas pareadas. Las
cinco configuraciones requieren veinte entrenamientos completos. V16b conserva solo la anchura
atómica compacta 64 y desactiva el contexto externo. V16c/d siguen incluyendo la referencia
anterior 16/64 hasta copiar explícitamente un finalista revisado. No heredan automáticamente
ninguna rama geométrica de V16a2;
una mejora observada sobre su referencia histórica 64/128 tendrá que comprobarse en la ruta compacta.

V16c reserva semillas nuevas para confirmar una configuración compacta 16/64; V16d es una
ficha de réplica del sistema congelado. La tasa de aprendizaje y la profundidad superficial
iniciales son **valores de referencia, no finalistas de HPO ya seleccionados**.
Antes de usarlos para confirmar o congelar una ganadora, copia la configuración completa revisada
de la etapa anterior, incluidas las fuentes y los valores numéricos, y registra la decisión en
el informe externo. No se importa una ganadora automáticamente ni se escoge la mejor semilla.
Las cinco recetas mantienen test sellado; redactar una decisión final no exige otro entrenamiento.

VN01 es una línea distinta de supervisión negativa en `experiments/vn1`, con la misma referencia
64/128 sin grafo residual, transferencia de rango 16, MAX y Heat6. `wisdom_vn1b.yaml` es un piloto
de cobertura e intensidad de diez épocas, no una aproximación al ranking final. Inspecciona
cobertura, actualizaciones activas, cocientes de gradiente y tiempo antes de fijar cortes, K e
intensidad. `wisdom_vn1.yaml` ejecuta ocho celdas × cuatro semillas comunes = **32 Runs**:
existencia regional positiva 0.3 o 0, cruzada con ausencia de minería, top-K global, top-K con
memoria y familias FGW actuales con memoria. R− retira solo ese prior positivo; BCE, gates,
dropout, decay y Heat6 siguen fijos.
Todas las recetas VN y VF desactivan los descriptores preentrenados de proteínas:
`external_model: null`, `external_mode: none` y `atomic_external_model: null` dentro de
`atomic_representation`. Esto desactiva el contexto de ESM-2, ESM-C y SaProt y la inicialización de
tablas con sus pesos; también quedan desactivados los descriptores atómicos congelados. Las tablas
de identidad de átomo y de residuo/átomo aprendidas desde cero siguen activas. Los archivos de
descriptores permanecen en el dataset, pero estas recetas no los exigen ni los cargan; no hace
falta repetir el preprocesado. Se busca examinar cada técnica sin que el contexto preentrenado de
proteínas domine la comparación, no maximizar las puntuaciones inmediatamente. Cambiar fuentes
de entrada cambia la identidad científica: inicia un estudio nuevo, no recuperes uno con ESM
activo usando el YAML modificado.
`wisdom_vn1a.yaml` conserva su nombre, pero ahora coincide con la matriz completa sin LLM de VN1.
No ejecutes ambos por defecto ni los interpretes como una comparación con/sin ESM.
`wisdom_vn1c.yaml` añade los controles puntual y softplus distribuida sin LLM (ocho Runs).
El antiguo piloto VN1a y los controles VN1b pasan a llamarse VN1b y VN1c; los resultados históricos
no se renombran. No recuperes un antiguo piloto VN1a usando la nueva matriz VN1a.
Los cortes 0.25/0.40, K=3 y coeficiente 0.1 son **propuestas de piloto sin
calibrar**. Fija decisiones comunes antes de la matriz, no coeficientes distintos tras ver cada resultado.

**N1 — regiones negativas actuales.** La etiqueta global de una proteína negativa aporta la
supervisión; la minería nunca carga su GT local. Al terminar cada época, una revisión determinista
evalúa todas las negativas de entrenamiento seleccionadas. El grafo superficial existente, filtrado
por radio, normales y cuerda, delimita componentes conectadas sobre el corte inferior. Una componente
puede activarse si su máximo alcanza el corte superior. Top-K es global para un Run, no por
proteína, lote o GPU. Los empates se resuelven por ID inmutable y menor índice de punto.
La primera época no tiene término selectivo; la revisión al terminar t prepara las máscaras de t+1.

Para una proteína b, z_bi es el logit operativo actual del punto i (después de Heat6 si está activo),
a_bi su peso proxy normalizado sobre **toda la proteína**, y u=log(U_s/(1−U_s)) el logit del
corte inferior U_s. H_b y Q_b son soportes disjuntos prioritarios y recordados, con coeficientes
λ_h y λ_m. La penalización de exceso por proteína es

$$
J_b=\lambda_h\sum_{i\in H_b}a_{bi}\max(0,z_{bi}-u)^2
   +\lambda_m\sum_{i\in Q_b}a_{bi}\max(0,z_{bi}-u)^2.
$$

En un lote B de proteínas muestreadas uniformemente, la suma de las negativas se divide por |B|π₋,
donde π₋ es la proporción negativa de toda la población de entrenamiento seleccionada. Así se estima
la media entre negativas sin cambiar la fuerza según la composición de cada lote. No se divide
cada región por su propia área. Las reducciones FP32 conservan el gradiente de los logits actuales;
los scores antiguos seleccionan soportes, nunca se convierten en targets. Un lote solo positivo
devuelve un cero diferenciable; una población sin negativas es inválida. Heat6 puede propagar
influencia fuera del soporte penalizado directamente. El control puntual iguala el área proxy
de cada proteína seleccionada, con un último punto fraccional; no afirma igual magnitud de gradiente.

La memoria empareja componentes **actuales** mediante solapamiento ponderado y condición de ancla
del máximo. Registra contracción, divisiones, fusiones, corrección, seguimiento perdido, expiración
y expulsión por presupuesto. Nunca penaliza máscaras archivadas. Por defecto los soportes débiles
reciben un cuarto del coeficiente fuerte, con hasta 4K soportes y cinco épocas desde su última
prioridad; el historial acotado conserva 10(K+4K) registros. Todo se configura en `negative_mining`.
Los checkpoints VN01 incluyen modelo, optimizador, scaler de precisión, mejor estado por G,
memoria, curvas y generadores de Python/NumPy/Torch/muestreo. Orden de lotes y semillas de workers
usan flujos separados, comunes a los controles actuales; no se afirma reproducir exactamente una
trayectoria histórica completa. La recuperación nativa de LF debe conservar el Run original;
un envío nuevo no es continuación. VN01 rechaza por ahora el promediado de pesos para no revisar
unos pesos y entrenar otros.

**N2 — prioridades diversas, no otra cabeza neuronal.** Cada época reconstruye el universo actual
de candidatas y recordadas, representantes geodésicos, distancias FGW y familias HDBSCAN nativas
de LF con matriz precomputada. FGW compara atributos y geometría interna mediante un mismo
acoplamiento de transporte. Los atributos son las curvaturas media/gaussiana existentes y química
explícita proyectada con un núcleo gaussiano, con medianas/IQR comunes de negativas de entrenamiento.
Logits, probabilidades y embeddings aprendidos no son atributos químicos. Máscaras por canal
distinguen soporte ausente de átomos/Cα/residuos conocidos.
Solo las observaciones válidas de entrenamiento ajustan las escalas; un atributo ausente usa
imputación centrada en cero con coordenadas explícitas de cobertura, no un cero físico sin marcar.
Un canal sin observaciones válidas detiene el preflight. No cambia el preprocesado ni las entradas
del predictor.

Hasta 64 representantes dispersos geodésicamente conservan masa proxy; sus caminos mínimos usan
el grafo disperso de la componente completa y una escala común de 10 Å, no el diámetro de cada
parche. Las matrices densas son **objetos acotados de comparación de regiones**, nunca grafos
de distancias de toda una proteína. Se auditan ambos sentidos de FGW, autocostes, marginales y
convergencia; la falta de convergencia recibe un intento predefinido con más iteraciones. Un fallo
numérico o de recursos provoca respaldo explícito a top-K esa época. La ausencia de POT es un
error de preflight, no respaldo silencioso; `./install.sh` y bootstrap instalan el extra
`optimal-transport` de LF. Las familias priorizan sus miembros elegibles más difíciles, permiten
un representante de ruido en la primera pasada y rellenan por dificultad. Todo ruido recupera
top-K. Los medoides describen familias; ni ellos ni los costes de transporte entran en backward.

**N3 — decisiones posteriores al entrenamiento.** `negative_calibration` actúa solo al restaurar
el checkpoint G. No modifica scores ni rankings G/S. Los parámetros tipados opcionales
`calibration_dataset` y `decision_dataset` aportan una reserva negativa y otra población de desarrollo
para el umbral F1. Se comprueba que sus IDs y grupos de fugas no coincidan con el dataset original,
incluida la pertenencia de test, sin leer sus etiquetas puntuales. Una negativa por grupo de reserva
aporta un máximo logit operativo. Independencia e intercambiabilidad siguen necesitando auditoría
del muestreo e historial: IDs diferentes no las demuestran. Sin reserva, `negative-admission.json`
indica indisponible y no informa una garantía formal. Nunca se reutilizan train, validación o test
silenciosamente como calibradoras.

F0 usa logit cero; F1 ajusta un umbral global por exactitud balanceada de desarrollo, desempatando
por especificidad; F2 usa p=(1+número de máximos calibradores ≥ máximo regional)/(n+1). Es un rango
de cola, no probabilidad de unión ni tasa de falsos descubrimientos. Su mínimo es 1/(n+1), por lo que
n≥19 es necesario solo para poder admitir al nivel 0.05. `formal: true` exige además
`history_attestation` no vacío; la interpretación sigue condicionada a intercambiabilidad. La
integración actual evalúa decisiones sobre validación para desarrollo, no sustituye una evaluación
final de test. Informa falsa admisión, sensibilidad, métricas binarias al umbral, área y Dice/IoU/
precisión/recall locales, conservando denominadores indefinidos como indisponibles. Cambiar umbrales
no mejora AUPRC/AUROC de scores que no han cambiado.

LF recibe curvas `train_negative_mining_*` y resúmenes finales `val_negative_admission_*`.
Los JSON por revisión conservan soportes, linajes y diagnósticos FGW; `evaluation.json` contiene
los resúmenes y el estado de admisión. El coste de revisión se registra separado del tiempo de
optimización/validación; `negative_mining_complete_epoch_seconds` incluye revisión y guardado de
continuación. Las pruebas verifican contratos de implementación, no mejor localización biológica.
Las hipótesis prospectivas completas y sus limitaciones permanecen en el informe.

### 4.7. Targets físicos VF

VF entrena el encoder atómico para reconstruir campos físicos alrededor de una proteína. Son
**objetivos, no descriptores de entrada ni etiquetas de unión**. El enriquecimiento opcional publica
el estricto `wisdom-dna-reduced-physics@1` desde `wisdom-dna-reduced-slae@1` y después el completado
acotado `wisdom-dna-reduced-physics@2` desde ese padre estricto. Ambos conservan cada miembro,
partición, etiqueta, NPZ base y archivo heredado, incluido SLAE. El predictor VF desactiva SLAE,
ESM y SaProt para que no permitan resolver la tarea sin aprender del grafo atómico.

| Configuración | Produce | Decisión siguiente |
|---|---|---|
| `experiments/preprocess/enrich_physics.yaml` | Volúmenes y bancos físicos del protocolo estricto histórico, `@1` | Recuperar los checkpoints originales antes de cambiar el protocolo científico |
| `experiments/preprocess/enrich_physics_bounded.yaml` | Completado acotado, `@2`, reutilizando campos estrictos readmitidos | Revisar átomos modelados, solapamientos, soporte y convergencia |
| `experiments/vf/wisdom_vf1.yaml` | Controles de reconstrucción física con cuatro semillas pareadas | Comprobar errores en proteínas no vistas y gradientes al encoder |
| `experiments/vf/wisdom_vf_fields.yaml` | Preentrenamiento por separado con potencial electrostático, LJ1 y LJ2 y las mismas cuatro semillas | Comparar cada campo con la rama de tres campos ya programada en VF1 |
| `experiments/vf/wisdom_vf_joint.yaml` | Entrenamiento conjunto desde cero con ocho pesos auxiliares, incluido cero | Comparar G proteica, localización, conflicto de gradientes y coste |

El piloto VF fija el candidato replicado de V16b sin ESM: una interacción SchNet, una
proyección de entrada superficial y seis bloques DiffusionNet. El YAML de entrenamiento
conjunto fija también su dropout (0,2), tasa (0,0007542867271657824) y decaimiento AdamW
(0,05) para los ocho pesos físicos. La fila de peso cero es el control funcional contemporáneo.
El preentrenamiento físico usa la misma ruta atómica y política de gates para permitir la
transferencia del checkpoint; no ejecuta los bloques superficiales y conserva un optimizador
propio para la reconstrucción. VN1c no acredita una ventaja frente a ausencia de minería con
el mismo código, por lo que el piloto VF la deja desactivada y aísla el objetivo físico.
El pico de 20,1 GiB de V16b se midió sin consultas físicas y no acota la memoria de VF.
Los YAML de entrenamiento VF seleccionan ahora el dataset físico estricto `@1`; si se adopta
el completado acotado `@2` tras revisarlo, hay que cambiar los tres juntos.
El estudio de campos individuales usa el mismo dataset estricto `@1` y la configuración de VF1c;
no necesita generar nuevos campos. Sus tres ramas reutilizan el control con los tres campos de
VF1c en lugar de entrenarlo otra vez. Compara el error en unidades originales del campo
supervisado y los gradientes del encoder; la pérdida de validación ponderada cambia de
significado según los campos y no sirve para ordenar las ramas.

Prepara el entorno y envía tú las etapas en ese orden:

```bash
lf clusters bootstrap citius-ctgpgpu16 --project .
lf validate experiments/preprocess/enrich_physics.yaml
lf explain experiments/preprocess/enrich_physics.yaml
lf run experiments/preprocess/enrich_physics.yaml --on citius-ctgpgpu16
lf run experiments/preprocess/enrich_physics_bounded.yaml --on citius-ctgpgpu16
lf run experiments/vf/wisdom_vf1.yaml --on citius-ctgpgpu16
lf run experiments/vf/wisdom_vf_joint.yaml --on citius-ctgpgpu16
lf run experiments/vf/wisdom_vf_fields.yaml --on citius-ctgpgpu16
```

No entrenes hasta aceptar la auditoría del enriquecimiento. La instalación local usa `./install.sh`;
`pip install -e '.[physics]'` aporta las API de Python, pero no el ejecutable nativo APBS. Conda
proporciona APBS, PDB2PQR/PROPKA y OpenMM. Se comprueban antes de procesar proteínas. No se repiten
selección, geometría ni extracción SLAE. La recuperación nativa compatible reutiliza checkpoints
validados por proteína; las referencias nuevas también usan la caché reconstruible de LF. El
diagnóstico se conserva en el checkpoint nativo `physics/last-report.json`, aunque se limpien los
artefactos de un intento fallido. Recupera el Work original, no envíes uno nuevo:

```bash
lf retry <failed-job-id> --config experiments/preprocess/enrich_physics.yaml --accept-code-change --accept-operational-changes --dry-run
lf retry <failed-job-id> --config experiments/preprocess/enrich_physics.yaml --accept-code-change --accept-operational-changes --apply
```

Revisa la simulación antes de aplicar. Aceptar código corresponde a esta corrección compatible,
no autoriza cambiar las entradas científicas. LF debe confirmar la identidad y restaurar las
dependencias; los archivos eliminados no se recuperan dando por supuesta una caché. El límite de
vóxeles del YAML controla recursos y no cambia la identidad de las referencias físicas.
StructuralEnrichment declara `physics_maximum_voxels` mediante la guarda pública de LF, que solo
permite aumentarlo. Los aumentos se admiten con aprobación operativa separada;
la simulación debe enumerar los checkpoints conservados del Work original. Esto **no**
autoriza cambiar la química al reintentar. Primero completa/recupera el estricto `@1`; después
ejecuta el YAML acotado como protocolo/versión nuevo. Su padre tipado permite revalidar y readmitir
campos estrictos sin repetir PDB2PQR/APBS/LJ. Las referencias no disponibles se reconsideran, no se
reutilizan sin más; cambiar herramientas o ecuaciones requiere nuevos cálculos. No se copian rutas
privadas de checkpoints.

La receta estricta permite ocho millones de vóxeles por rejilla APBS; la acotada permite
24 millones para cubrir proteínas alargadas y sus auditorías con rejilla más fina y caja ampliada,
dentro de la asignación existente de 96 GiB. Es un límite de asignación, no una orden de agrandar
todas las rejillas ni una garantía de memoria. El log identifica cada rejilla y su número de
vóxeles. Si el Work acotado alcanza el límite antiguo, recupera **ese mismo Work acotado**, no el
productor estricto:

```bash
lf retry <bounded-job-id> --config experiments/preprocess/enrich_physics_bounded.yaml --accept-operational-changes --dry-run
lf retry <bounded-job-id> --config experiments/preprocess/enrich_physics_bounded.yaml --accept-operational-changes --apply
```

Añade `--accept-code-change` solo si también aplicas una corrección de código compatible revisada.
La recuperación revalida las referencias y bancos guardados y calcula lo pendiente; no reinicia
selección, geometría, extracción SLAE ni cálculos físicos correctos. Los fallos reales de recursos
o herramientas siguen siendo errores, nunca química no disponible ni campos silenciosamente
calculados con menor resolución.

**Referencia y unidades.** PDB2PQR/PROPKA prepara hidrógenos y protonación a pH 7. Los átomos
pesados retenidos deben conservar identidades, elementos y coordenadas, con tolerancia de 0,002 Å
por redondeo PDB. El estricto `@1` rechaza átomos pesados modelados; el acotado `@2` admite completado
limitado con los criterios siguientes y las mismas ecuaciones físicas. Una preparación no
admisible deja VF no disponible sin alterar el grafo. La proteína conserva
partición, SLAE y geometría, con un motivo explícito y banco físico vacío; no aporta pérdida física
ni una métrica de error cero. Los desplazamientos de átomos, fallos de herramientas desconocidos,
límites de recursos y desacuerdos con la auditoría independiente LJ sí bloquean la publicación.
Los hidrógenos preparados y los átomos modelados admitidos
son fuentes físicas, nunca nodos del modelo. ADN funcional, Zn, agua y ligandos excluidos no son
fuentes. APBS resuelve el piloto declarado de Poisson–Boltzmann lineal con solvente continuo:
298,15 K, permitividades interior/exterior 2/78,5, radio de solvente 1,4 Å e iones +1/−1 a
0,15 mol/L cada uno. El potencial permanece en kBT/e, con constante de Boltzmann kB, temperatura T
y carga elemental e; no se inserta una sonda cargada real. Las dos sondas neutras Lennard–Jones
usan tipos reales AMBER14 `ALA:HB1`/`protein-HC` y `ALA:CB`/`protein-CT`. Diámetros y profundidades
de pozo se leen del XML instalado; cambian ambos, no solo el tamaño. Las distancias pasan de Å a
nm y las energías permanecen en kJ/mol. La combinación usa media aritmética de diámetros y
geométrica de profundidades. Por defecto, LJ suma toda la proteína preparada finita. Cada proteína
debe superar una comparación independiente con un grupo de interacción proteína–sonda de OpenMM,
con tolerancias absoluta/relativa de 1e-5.

**Volúmenes y máscaras.** La separación nominal es 1 Å y el margen de caja 12 Å; se guardan
origen, separaciones y forma reales compatibles con multigrid. El refinamiento a 0,75 veces la
separación nominal y una caja mayor comparan las mismas posiciones físicas. La caja mayor conserva
la separación fina original. El acuerdo numérico no demuestra exactitud del modelo físico.
`potential_failure_policy: mask` conserva valores brutos en todas las posiciones, excluyendo de
supervisión las inestables/no comunes. El piloto exige cobertura de potencial de 95% global y
80% en cada región presente, con tolerancia `absolute_error <= 0.25 + 0.1 * abs(reference)` en
kBT/e. `fail` exige acuerdo en todo el dominio. Son umbrales piloto configurables, no tolerancias
universales validadas. Si la cobertura queda por debajo de cualquiera de los umbrales con `mask`,
se descarta **todo el canal de potencial** de esa proteína; se conservan valores brutos y auditoría
fallida, pero siguen disponibles los canales LJ auditados por separado. `fail` bloquea la
publicación. El YAML estricto permite ocho millones de vóxeles por rejilla, con cuatro solvers de un hilo
dentro de 96 GiB; no engrosa la rejilla silenciosamente. La normalización usa solo targets válidos
de entrenamiento. La validación excluye proteínas no disponibles de la media de pérdida física e
informa su número; se falla si un campo seleccionado no tiene targets en todo entrenamiento o si
no hay targets válidos de validación. Revisa soportes por partición y campo antes de interpretar
VF: la falta de datos dependiente de la química puede sesgar el subconjunto supervisado.

**Estructuras incompletas.** Que falten coordenadas no invalida automáticamente una estructura
depositada: el experimento puede no resolver átomos flexibles o residuos enteros. Un campo de
fuerzas molecular necesita, sin embargo, una estructura químicamente especificada. PDB2PQR
normalmente reconstruye algunos átomos pesados al prepararla; `--noopt` no desactiva esa reparación.
El protocolo estricto de átomos observados de WISDOM no trata esas coordenadas nuevas como
observaciones. Los tres campos actuales comparten esa preparación: si no es admisible, quedan
no disponibles **los tres**, no solo APBS. En cambio, una comparación de cobertura APBS fallida
desactiva solo el potencial si las auditorías independientes LJ se superan. Los informes separan
estos casos, conservan las diferencias de identidad atómica disponibles sin truncarlas y muestran
el soporte por campo y partición.

Una conformación alternativa es una colocación observada diferente de algunos átomos. Elegirlos
individualmente puede mezclar alternativas en un archivo base antiguo. Antes de PDB2PQR, WISDOM
comprueba los oxígenos terminales observados llamados `OXT`: debe existir su carbono carbonílico,
la distancia C--OXT debe medir entre 0,9 y 2,2 Å, y ese carbono no puede continuar simultáneamente
un enlace peptídico al siguiente residuo observado a 0,9--1,9 Å. Una contradicción deja los tres
campos VF no disponibles, con la identidad atómica y las distancias medidas en la auditoría. No
elimina la proteína, cambia sus coordenadas base o SLAE, ni aborta las demás proteínas. WISDOM no
renombra fragmentos, elimina un oxígeno observado ni reconstruye el padre silenciosamente para
forzar la preparación. Las pérdidas inexplicadas, desplazamientos o cambios de elemento de átomos
observados siguen siendo errores. Esta comprobación concreta no garantiza la corrección química
general; cambiar la conformación del padre requiere un dataset nuevo.

`heavy_atom_policy: bounded` aplica estos **criterios piloto de admisión** deliberadamente
limitados. Son guardas configurables, no tolerancias universales validadas experimentalmente:

| Configuración y valor seleccionado | Significado |
|---|---|
| `repair_maximum_atoms: 32` | Máximo total de átomos pesados modelados nuevos, incluido OXT terminal |
| `repair_maximum_fraction: 0.05` | Los añadidos divididos por los pesados observados tampoco pueden superar el 5% |
| `repair_maximum_per_residue: 5` | Ningún residuo recibe más de cinco átomos pesados modelados |
| `repair_maximum_residue_fraction: 0.1` | Como máximo el 10% de los residuos observados necesitan completado |
| `repair_minimum_nonbonded_fraction: 0.65` | Rechazar solapamientos graves por debajo del 65% del diámetro LJ combinado del campo de fuerzas |
| `repair_exclusion_radius: 3.0` | Excluir consultas de entrenamiento a menos de 3 Å de cualquier átomo pesado modelado |

Los límites de cantidad se aplican simultáneamente. N/CA/C/O del esqueleto deben existir, las
distancias C--N peptídicas observadas consecutivas deben estar entre 0,9 y 1,9 Å, no se añaden
residuos ni se mueven átomos observados. Modelar OXT exige que `sequence_context`, vinculado a la
base, confirme el último token de la cadena depositada; no basta el último residuo observado antes
de una cola sin resolver. OpenMM debe parametrizar la estructura preparada completa. Los enlaces
pesados modelados deben medir entre 0,9 y 2,2 Å; la prueba de solapamientos excluye pares enlazados
y vecinos a través de dos enlaces. Siguen desactivados el debumping y la optimización lateral de
PDB2PQR que podrían mover átomos observados: las colocaciones modeladas inadecuadas se rechazan,
no se ocultan desplazando la geometría experimental.

Se archivan los campos calculados completos. La exclusión del banco usa coordenadas float32 como
las que ve el decodificador, evitando supervisar el detalle inmediato de un átomo no observado.
No acota efectos electrostáticos lejanos ni certifica una conformación reparada. La auditoría
conserva cantidades, identidades, geometría, evidencia de secuencia y hashes de reutilización;
el soporte se informa por campo/partición. Las fuentes reparadas nunca son descriptores ni nodos
del encoder. Enmascarar protege almacenamiento y optimización, no demuestra cobertura VF suficiente
o representativa. Revisa por separado proteínas observadas y reparadas antes de extraer conclusiones
biológicas. La evidencia histórica está en `benchmarks/physics_preparation_audit.json`.
El cálculo LJ completo usa bloques acotados de 32 consultas, con el mismo orden atómico y
reducciones float64. Un microbenchmark sintético de CPU midió una aceleración de 2,38× con
valores/máscaras idénticos bit a bit (`benchmarks/physics_lj_runtime.json`); no estima la aceleración
del pipeline completo ni del cluster.

La distancia firmada a esferas de van der Waals de átomos pesados define interior y frontera. La
conectividad por vóxeles libres para la sonda de solvente identifica exterior alcanzable y huecos
geométricos cerrados, no agua observada ni bolsillos funcionales. Los canales alcanzables comparten
estratos exteriores. Se usan seis vecinos por defecto; 26 es opcional. La estratificación propone
40% frontera, 25% exterior cercano, 20% hueco cerrado y 15% otro exterior, redistribuyendo estratos
ausentes sin duplicados. Cada sonda LJ enmascara por separado solapamientos y energías de valor
absoluto superior a 10000 kJ/mol, conservando la banda repulsiva finita. Los ceros almacenados
bajo máscara no son observaciones de energía cero.

`physical_reference` es un NPZ vinculado al hash base con campos completos, máscaras, errores
numéricos, distancia/accesibilidad, correspondencia de átomos preparados, cargas/parámetros LJ,
PQR, PDB preparado, entrada APBS/OpenDX, unidades, versiones y auditorías. `physical_queries`
guarda por separado hasta 16384 posiciones únicas `[Q,3]`, targets/máscaras booleanas `[Q,3]`,
región/distancia/accesibilidad y direcciones de cuadrícula. Todos los átomos retenidos a menos de
12 Å son candidatos: `offsets[Q+1]` delimita listas contiguas `atom_ids`, sin límite top-K.
Son legítimas las vecindades vacías.

Solo entra geometría de consulta al decodificador. Suma mensajes que combinan estados atómicos
y bases de distancia; tres cabezas predicen potencial/LJ1/LJ2. Para aproximar largo alcance, el
potencial recibe contexto atómico medio y distancia al centroide atómico, no una solución exacta
APBS/Coulomb. No hay matriz densa consulta–átomo ni cambios de pooling funcional o encoder
superficial. Las cabezas condicionadas, encoder congelado, estados nulos/permutados, decodificador
pequeño y ausencia de mensajes son controles explícitos.

**Optimización.** `physics: null` no cambia Training. `mode: pretrain` ejecuta solo el encoder
atómico y las cabezas físicas/de control, sin usar etiquetas funcionales, GT local ni test. Las
particiones heredadas train/validación, separadas por grupos, definen un split físico del mismo
dominio, no un corpus externo independiente. La loss de validación física selecciona su época.
`mode: joint` añade errores físicos a la pérdida funcional, pero el checkpoint sigue usando solo G
proteica.

Para el campo s, Ls promedia error normalizado solo en sus consultas válidas. El potencial conserva
signo; LJ usa `asinh(U/u0)`, con energía U y u0 igual a la mediana de energía absoluta válida de
entrenamiento, limitada por debajo a 1e-6 kJ/mol. Medias y desviaciones también usan solo train.
El objetivo es `L_total = L_WISDOM + sum_s lambda_s * Ls`. Elige `objective.weight` global o
`objective.weights` individuales, nunca ambos. Por defecto se usa Huber con beta 1; MSE es opcional.
Se muestrean 512 consultas por proteína/actualización; validación usa 4096 consultas fijas y la
evaluación final cubre todo el banco. LF registra MAE/RMSE en unidades originales,
Pearson/Spearman espacial, acierto del signo fuera de ±0,1 kBT/e, soportes regionales y gradientes
al encoder. Las correlaciones indefinidas quedan no disponibles, no cero. Son métricas de
reconstrucción, no de exactitud biológica de unión.
`val_physics_unweighted_loss` excluye los coeficientes auxiliares y sigue siendo informativa en
la fila de presión cero, a diferencia de la loss ponderada. Los pesos individuales omitidos son cero.

En el entrenamiento conjunto, `precision: auto` mantiene las capas densas en BF16 en GPU
compatibles. Las derivadas superficiales dispersas usan FP32; la retropropagación del optimizador
y la comparación de gradientes funcionales/físicos se ejecutan fuera del autocast del forward
(precisión mixta automática). La comparación deriva el mismo grafo conservado sin rellenar los
gradientes del optimizador ni repetir el forward. Así se evitan operaciones dispersas CUDA de
retropropagación no soportadas en BF16 sin desactivar diagnósticos ni cambiar pérdidas.
El piloto exclusivamente físico sigue usando FP32 explícitamente.

El preentrenamiento publica `physical-encoder.pt`, normalización y evaluación; la continuación
nativa conserva optimizador, RNG, época y mejores pesos. Enlaza un archivo revisado con
`physics_checkpoint: {file: path/to/physical-encoder.pt}` para transferir una configuración atómica
compatible. No se transfieren cabezas funcionales/superficiales. Las físicas se transfieren con
normalización train coincidente; `transfer_decoder: false` las reinicia. Se contrastan grupos/hashes
train de origen contra validación de destino. `freeze_encoder: true` desactiva dropout atómico y
puertas estocásticas. `pretraining_task: masked_element` reconstruye categorías atómicas ocultas
como control no físico real, con presupuesto fijo de épocas; transfiere la última época en lugar
de seleccionar por cabezas físicas sin entrenar.

El YAML conjunto es un **piloto auxiliar desde cero**, no la conclusión final VF1f de la memoria.
Los checkpoints físicos revisados y la arquitectura que termine eligiéndose en V16 deben fijarse
explícitamente antes de comparar T0–T4. No se inventan ganadores ni entrenamientos anidados.

## 5. Experimentos y desarrollo

### 5.1. Entrenamiento y visores de proteínas

Cada YAML de entrenamiento es un único Work `wisdom.Training.Training`. `with` aporta los parámetros
científicos; `seeds`, `sweep` o `search` describen repeticiones/candidatos; `resources` y, opcionalmente,
`execution` gestionan la admisión. `objective` define lo que optimiza LF. No deduzcas las reservas
de GPU ni la concurrencia únicamente por el tamaño del modelo. LF gestiona ejecución, cachés,
resultados y recuperación; WISDOM implementa las transformaciones científicas y el bucle PyTorch.

`data_workers` controla los subprocesos que decodifican arrays mientras la GPU procesa el lote
anterior; cero realiza la lectura en el proceso de entrenamiento. WISDOM inicializa los workers
de entrenamiento, validación y revisión con la estrategia `file_system` de PyTorch. Esta envía
tensores de CPU mediante memoria compartida con nombre (RAM, no una copia del dataset en disco),
sin mantener un descriptor abierto por tensor. Un descriptor es un identificador de recurso del
sistema operativo; un límite bajo por proceso puede causar `Too many open files` en lotes con
muchos operadores por proteína y representaciones externas, aunque los NPZ estén bien cerrados.
El entrenamiento anticipa un lote por worker. Esta política no cambia valores ni semillas, no
eleva límites del sistema y deja la asignación y limpieza a `torch_shm_manager`, de PyTorch.
La memoria compartida debe seguir teniendo capacidad para los tensores anticipados; no es ilimitada.

Antes de la primera época, WISDOM valida cada partición solicitada y sus archivos estructurales
auxiliares. Los logs muestran progreso cada 25 proteínas seleccionadas y el tiempo de cada
partición. La evaluación superficial reutiliza esas entradas validadas en lugar de reconstruir
dos veces las mismas entradas de validación/test. Los controles de contexto atómico descomprimen
solo los campos atómicos, pero siguen verificando la suma de comprobación del archivo base
completo y reconstruyendo todos los campos del contexto. No se desactiva ninguna validación ni se
reutiliza entre entrenos distintos. LF registra `dataset_setup_seconds`,
`representation_fit_seconds` (cuando corresponde) y `startup_seconds`; esta última mide la
preparación de la aplicación hasta estar lista para entrenar, no la espera en cola, instalación
del entorno ni transferencia remota de datos. Una GPU inactiva durante estos controles de
CPU/archivos no demuestra un bloqueo. Tampoco la etiqueta `running` demuestra progreso científico:
contrasta la antigüedad de la observación de LF con los recuentos de entrenos completados y el
resultado final guardado de la ejecución.

```bash
lf validate experiments/v1/wisdom_v1a.yaml
lf explain experiments/v9/wisdom_v9d.yaml
lf run experiments/v9/wisdom_v9d.yaml --dry-run
lf run experiments/v9/wisdom_v9d.yaml --on YOUR_CLUSTER
```

Configura los visores finales en `with.visualization`: `mode` admite `none`, `report`, `viewer` o
`full`. `none` no genera visores ni pestañas vacías; `report` incrusta inspectores; `viewer` añade
HTML/PLY; `full` añade NPZ de predicción. `content: predictions` conserva probabilidades, logits y
GT; las opciones más completas incluyen otros canales. `maximum_proteins` y `maximum_points`
afectan solo a la representación, no a la cobertura científica de las métricas. Los comentarios
de los YAML detallan los valores predeterminados, límites de tamaño y criterios de inclusión.

Los visores activados se generan una vez desde el mejor checkpoint restaurado de ese Run, no cada
época ni después de una selección oculta de la mejor semilla. `surface_metrics_interval: 0` calcula
diagnósticos locales al final; un N positivo también los mide cada N épocas de validación. Los
informes aparecen como artefactos y pestañas WISDOM de LF. Un revisor como
`experiments/v5/visualization/wisdom_v5a.yaml` representa checkpoints importados sin reentrenar.
Selecciona ejecución de origen, semilla guardada y filtro opcional de parámetros en ese YAML. Si
falta un checkpoint, se necesita recuperación nativa, no un entrenamiento de reemplazo automático.

### 5.2. Organización del código y verificación

Mantén una clase por archivo Python, incluidos los enums. Agrupa por responsabilidad, no por número
de experimento:

| Paquete | Responsabilidad |
|---|---|
| `models/atomic`, `residue`, `surface`, `communication` | Codificación atómica/de residuos, superficial y transferencia entre representaciones |
| `models/pooling`, `heads`, `refinement` | Agregación de proteína, cabezas y refinamiento espacial de evidencia |
| `models/physics`, `preprocessing/common/structure/assets/physics` | Decodificadores de consultas, objetivos/transferencia física y referencias auditadas |
| `models/initialization`, `optimization`, `losses`, `gating`, `architecture` | Inicialización, políticas de optimización, objetivos y controles arquitectónicos explícitos |
| `data`, `evaluation`, `analysis` | Lectura/agrupación del dataset, métricas científicas y descripciones para LF |
| `preprocessing/dna`, `preprocessing/zinc` | Evidencia, selección, orquestación y anotaciones de cada tarea |
| `preprocessing/common/independence`, `sampling`, `clustering`, `auditing` | Etapas científicas compartidas de selección |
| `preprocessing/common/structure/io`, `geometry`, `storage`, `assets`, `viewer` | Coordenadas, construcción geométrica, persistencia validada, recursos opcionales e inspector común |
| `utils/structure` | Objetos estructurales reutilizados: proteína → cadena → residuo → átomo |
| `visualization`, `interpretability` | Revisión/informes de checkpoints y análisis posterior de conceptos |

Permanecen las rutas públicas de los Works y `models/WisdomV1.py`, `WisdomV2.py`, `WisdomV3.py`.
Los imports internos siguen los paquetes nuevos; no hay implementaciones duplicadas ni una colección
de archivos que solo reenvían imports. Selection y Preprocessing conservan su orquestación comentada,
legible como pseudocódigo científico. Las etapas sin estado siguen siendo funciones, no clases
de servicio artificiales.

Después de modificar código:

```bash
ruff check .
mypy src/wisdom
pytest -q
lf validate experiments/preprocess/dna/dna_preprocess.yaml
lf explain experiments/preprocess/dna/dna_preprocess.yaml
lf run experiments/preprocess/dna/dna_preprocess.yaml --dry-run
lf validate experiments/preprocess/dna/validate_dna.yaml
lf validate experiments/v1/wisdom_v1a.yaml
lf validate experiments/v2/wisdom_v2.yaml
```

Las pruebas comprueban contratos, comportamiento numérico y empaquetado; no demuestran mejoras
científicas. Las pruebas de navegador, pesos externos y datos de producción pueden necesitar
entradas locales opcionales. Guarda mediciones en `benchmarks/`, modelos descargados en
`data/models`, resultados bajo gestión de LF y discusión científica en la memoria. No edites
`build/lib`: es salida generada del empaquetado, no otro árbol de código fuente.

## 6. Referencias

La memoria contiene la bibliografía completa y sus matices. Referencias principales de implementación:

- [RCSB PDB](https://www.rcsb.org/) y [PDBx/mmCIF](https://mmcif.wwpdb.org/): coordenadas depositadas y sus campos.
- [Gemmi](https://gemmi.readthedocs.io/): lectura de estructuras y ensamblajes.
- [MMseqs2](https://doi.org/10.1038/nbt.3988) y [Foldseek](https://doi.org/10.1038/s41587-023-01773-0): búsqueda de secuencias y estructuras.
- [HDBSCAN](https://doi.org/10.1007/978-3-642-37456-2_14): agrupación por densidad; el ruido es un resultado legítimo.
- [DiffusionNet](https://doi.org/10.1145/3507905): aprendizaje superficial geométrico; WISDOM usa su adaptación documentada, no afirma una implementación idéntica.
- [Gates Hard-Concrete](https://arxiv.org/abs/1712.01312): regularización L0 aproximada y diferenciable.
- [SchNet](https://doi.org/10.48550/arXiv.1706.08566): interacciones con filtros continuos de distancia; WISDOM usa la capa pública de LF con sus adaptadores y controles declarados.
- [Versión pública fijada de SLAE](https://github.com/ProteinDesignLab/SLAE/tree/078b94e436b8eceb02251a254c3c224358ec6ee2): encoder atómico congelado opcional, vocabulario atom37 y checkpoint base; se desconoce el solapamiento con el benchmark.
- [Multiprocesamiento de PyTorch](https://docs.pytorch.org/docs/2.11/multiprocessing.html#sharing-strategies): estrategias para compartir tensores de CPU y limpieza nativa de memoria compartida.
- [Fused Gromov–Wasserstein](https://proceedings.mlr.press/v97/titouan19a/titouan19a.pdf): comparación conjunta de atributos y relaciones; VN01 usa el solver local de LF basado en POT, no un óptimo global certificado.
- [Valores p conformales para observaciones atípicas](https://doi.org/10.1214/22-AOS2244): principio de calibración por rangos finitos; su interpretación para negativas de VN01 sigue condicionada al muestreo e independencia.
- [PDB2PQR/PROPKA](https://pdb2pqr.readthedocs.io/en/latest/using/index.html): preparación y condiciones declaradas de protonación.
- [Coordenadas ausentes en RCSB](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/missing-coordinates) y [reparación de átomos pesados en PDB2PQR](https://pdb2pqr.readthedocs.io/en/latest/api/biomolecule.html): átomos sin resolver y diferencia entre geometría observada y completado modelado.
- [Mallas APBS](https://apbs.readthedocs.io/en/stable/using/input/old/elec/mg-auto.html) y [unidades del potencial](https://apbs.readthedocs.io/en/latest/using/input/old/elec/write.html): enfoque, cuadrículas numéricas y salida kBT/e.
- [Grupos de interacción OpenMM](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomNonbondedForce.html): referencia LJ proteína–sonda independiente.
