[English](README.md)

# WISDOM V6: regularizaciones débiles aisladas

- [1. Ejecutar e inspeccionar](#1-ejecutar-e-inspeccionar)
- [2. Comparación congelada y diseño experimental](#2-comparación-congelada-y-diseño-experimental)
- [3. Pérdidas exactas e interpretación](#3-pérdidas-exactas-e-interpretación)
- [4. Mezclas exploratorias y revisión de checkpoints](#4-mezclas-exploratorias-y-revisión-de-checkpoints)

## 1. Ejecutar e inspeccionar

V6a–V6g son estudios aislados independientes con una Work, no steps ni ganadores copiados.
V6a/c/d/f conservan sus parrillas originales y añaden valores altos en ramas concretas; amplía los
estudios completados mediante LF, sin reiniciar los controles históricos. V6h es un estudio
exploratorio ejecutable informado por V6a–V6f; V6g sigue pendiente, no se supone favorable.

Usa el contenido exacto del benchmark V5c:
`sha256:ac1745504422cf60b84e57b90efc99206dea039dc36d0c1359782062cb10e202`.
Una versión con igual nombre y distinto contenido no es equivalente. Comprueba el Registry
del destino: validar el YAML local no demuestra equivalencia del contenido remoto.

```bash
lf datasets show wisdom-dna-reduced@6 --on citius-ctgpgpu12
lf validate experiments/v6/wisdom_v6c.yaml
lf explain experiments/v6/wisdom_v6c.yaml
lf run experiments/v6/wisdom_v6c.yaml --on citius-ctgpgpu12 --dry-run
lf run experiments/v6/wisdom_v6c.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6d.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6e.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6f.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6g.yaml --on citius-ctgpgpu12
lf validate experiments/v6/wisdom_v6h.yaml
lf explain experiments/v6/wisdom_v6h.yaml
lf run experiments/v6/wisdom_v6h.yaml --on citius-ctgpgpu12 --dry-run
lf results list
lf results analyze wisdom-v6f
lf results report wisdom-v6f --output v6f-report.html
lf run experiments/v6/visualization/wisdom_v6f.yaml
```

Cambia el cluster cuando corresponda. Importa evidencia y checkpoints remotos con `lf import`
antes de revisar localmente. El selector `source_execution: wisdom-v6f` debe identificar una
única ejecución; si es ambiguo, usa su ID real `execution-…`. No se elige implícitamente la
última ejecución ni se reentrenan modelos ausentes.

## 2. Comparación congelada y diseño experimental

La configuración coincide con V5c y el parent científico de los estudios aislados V6. No se reabre la
búsqueda de inicialización, optimizador, encoder, hiperparámetros del pooling ni dataset.

| Elemento | Valor fijo |
|---|---|
| Dataset / generación / vista | `wisdom-dna-reduced@6` / `model_version=2` / `full` |
| Inicialización / optimización / cambio arquitectónico / cabeza | `baseline` / `baseline` / `baseline` / `single` |
| Anchura oculta / elemento / residuo | 128 / 32 / 64 |
| Profundidad atómica / proyección / superficie | 2 / 3 / 4 |
| Vecinos atómicos / transferencia / modos espectrales | 8 / 16 / 32 |
| Radio de transferencia / dropout | 4 Å / 0.2 |
| Tasa AdamW / decaimiento / coeficiente de gates | 0.001 / 0.0005 / 0.04 |
| Lote / límite de épocas / paciencia G / delta mínimo | 16 / 500 / 30 / 0.001 |
| Precisión / workers de datos | `auto` / 4 |
| Evaluación superficial / intervalo | activada / cada época |
| Fidelidad por borrado / evaluación test | desactivada / desactivada |
| MAX | control de puntos; sin campos LSE/Attention activos |
| LSE | área, beta fijo 0.25 |
| Attention | área, scorer simple desde embeddings, anchura 256 |
| Perfil débil | `custom`; una familia de coeficientes por estudio aislado |
| Semillas | 4, 7, 32, 54 |
| Objetivo / modo / rango / margen práctico | `val_wisdom_hpo_score` / max / [0,1] / 0.015 |
| Reserva / presupuesto de despacho | Las GPU son operativas y se leen de cada YAML; h solicita 3 GPU; 36 CPU, 96 GiB, 168 h / 168 h |

G mide clasificación de proteínas y S localización superficial en positivas. Sus definiciones,
soportes y la utilidad de desarrollo W están en [README, sección 5.5](../../README.es.md#55-entrenamiento-evaluación-y-artefactos).
W no cambia; **solo G selecciona el checkpoint de cada Run**. El GT local de contacto entra
en evaluación/utilidad de desarrollo, nunca en pérdidas ni selección del checkpoint. Test sigue sellado.

| Estudio / regularización | Hipótesis / patología que aborda | Parámetros activos | Interacción con Heat | Trials / Runs |
|---|---|---|---|---:|
| V6a negativo | Eliminar masa/picos positivos espurios en proteínas negativas | parrilla original más puntos altos selectivos LSE/Attention | fijo 6 Å | 23 / 92 |
| V6b existencia (congelado) | Exigir al menos un pico positivo en cada proteína positiva | lambda 0, .03, .10, .30 | fijo 6 Å | 12 / 48 |
| V6c positivo regional | Exigir una región positiva, no solo un pico puntual | parrilla original más ampliaciones LSE/MAX; longitud 2.5 Å | difusión extra tras Heat6 | 19 / 76 |
| V6d ranking regional | Separar máximos regionales positivos/negativos | parrilla original más ampliaciones LSE/Attention; longitud 2.5 Å; margen .5 | puntuación regional tras Heat6 | 20 / 80 |
| V6e cardinalidad | Evitar masa probabilística positiva insuficiente; probar dependencia del prior | lambda 0, .03, .10, .30; mínimo activo .005, .010, .020; sin máximo | fijo 6 Å | 30 / 120 |
| V6f TV | Penalizar saltos locales sin borrar necesariamente las fronteras | parrilla original más ampliaciones LSE/none y MAX/Heat | none / Heat6 | 31 / 124 |
| V6g Dirichlet | Penalizar variaciones probabilísticas de alta frecuencia | lambda 0, .001, .010, .050, .100 | none / Heat6 | 30 / 120 |
| V6h mezclas exploratorias | Separar interacciones, redundancia y estrés con todas las familias | 18 combinaciones explícitas por pooling; coeficientes provisionales | fijo 6 Å; TV/Dirichlet excluyentes | 54 / 216 |

Los YAML aislados actuales declaran **165 Trials / 660 Runs en V6a–V6g**. Incluyen los 30 / 120
añadidos selectivos de a/c/d/f y los cinco valores ya escritos en V6g (30 / 120).
Son recuentos del diseño, no ejecuciones completadas. V6h añade por separado 54 / 216.
El coeficiente cero de cardinalidad no tiene un mínimo activo: hay un control, no tres.
La longitud Heat no aparece en Trials identidad. Las condiciones nativas `when` también omiten
campos de pooling inactivos. Los barridos completos desactivan pruning adaptativo y seed racing,
pero la paciencia de G sigue deteniendo la optimización de cada Run. El límite de 168 horas puede
dejar cobertura incompleta; comprueba todos los Runs requeridos antes de cerrar el estudio.

## 3. Pérdidas exactas e interpretación

Una proteína es un conjunto de puntos superficiales: el entrenamiento conoce su etiqueta binaria,
no qué puntos unen ADN. Sean B+ y B− las proteínas positivas y negativas del lote; P_b sus puntos;
l_p el **logit operacional** del punto p, q_p = sigmoid(l_p) su probabilidad y a_p su área
representada dividida por el área total de su proteína: los a_p suman uno por proteína.
Define softplus(x) = log(1 + exp(x)) y [x]+ = max(0,x). El mismo l operacional entra en
pooling y pérdidas débiles; los logits crudos de la cabeza quedan como diagnóstico.
Cada término dependiente de una clase aporta cero diferenciable si esa clase falta.

El prior negativo promedia BCE negativa puntual ponderada por área entre proteínas negativas.
Existencia exige un pico positivo en cada proteína positiva:

```math
L_{neg}=\frac{1}{|B_-|}\sum_{b\in B_-}\sum_{p\in P_b}a_p\operatorname{softplus}(l_p),
\qquad
L_{exist}=\frac{1}{|B_+|}\sum_{b\in B_+}\operatorname{softplus}(-\max_{p\in P_b}l_p).
```

Con MAX, existencia repite la BCE positiva global: cambia el peso relativo de las clases, no la
restricción de localización. LSE/Attention por área no tienen esa identidad. Una presión negativa
fuerte puede eliminar indirectamente evidencia positiva real por compartir parámetros: revisa ambas clases.

**Términos regionales.** Phi_b contiene los modos superficiales almacenados, ortonormales con masa;
M_b es la diagonal de áreas y Lambda_b sus autovalores no negativos en Å inversos al cuadrado.
La difusión usa la base truncada existente, no un nuevo preprocesado. Con longitud r = 2.5 Å produce
u_b = Phi_b exp(−r² Lambda_b) Phi_bᵀ M_b l_b. Sea R_b = max(u_b).
La pérdida positiva regional aplica existencia a estos máximos; ranking compara **cada par
positivo-negativo del lote**, con margen de logits fijo delta = .5:

```math
L_{regional+}=\frac{1}{|B_+|}\sum_{b\in B_+}\operatorname{softplus}(-R_b),
\qquad
L_{rank}=\frac{1}{|B_+||B_-|}\sum_{b\in B_+}\sum_{c\in B_-}[\delta-R_b+R_c]_+.
```

Ranking vale cero en lotes de una sola clase. En c/d la difusión regional sigue al Heat6
operacional: su aportación puede ser redundante. No es un segundo encoder aprendido ni un objetivo
supervisado con anotaciones espaciales.

**Cardinalidad.** La masa esperada m_b = sum(a_p q_p) es una probabilidad ponderada por área,
no la fracción que supera un umbral binario. Con mínimo mu y sin máximo, la pérdida actual es:

```math
L_{card}=\frac{1}{|B_+|}\sum_{b\in B_+}[\mu-m_b]_+^2,
\qquad m_b=\sum_{p\in P_b}a_p q_p.
```

Vale cero cuando se alcanza el mínimo. Con el mayor mínimo .02, su máximo sin ponderar es .0004;
con lambda .30 aporta como máximo .00012. Estos valores conservadores pueden resultar inactivos
o débiles: comprueba la pérdida registrada y la masa antes de interpretar un efecto nulo como
fracaso de la idea. No se cambia a escondidas su escala ni se añade un máximo.

**TV no es Heat.** E es el conjunto con multiplicidades de entradas dirigidas válidas en la tabla
acotada de vecinos superficiales. TV promedia diferencias absolutas de probabilidad, sin ponderar por área o distancia:

```math
L_{TV}=\frac{1}{|E|}\sum_{(i,j)\in E}|q_i-q_j|.
```

Una tabla vacía aporta cero diferenciable. La penalización L1 puede favorecer regiones homogéneas
y permitir fronteras más marcadas que un suavizado cuadrático, pero no lo garantiza: pueden
desaparecer zonas de unión pequeñas reales. El promedio del lote pondera las proteínas según
su número de vecinos válidos, no por igual como los términos anteriores.

**Dirichlet tampoco es Heat.** Calcula los coeficientes c_b = Phi_bᵀ M_b q_b.
La energía normalizada actual, promediada por igual entre proteínas, es:

```math
L_D=\frac{1}{B}\sum_{b=1}^{B}
\frac{\sum_k\lambda_{bk}c_{bk}^2}
     {\max(\sum_k c_{bk}^2,\epsilon)}.
```

B es el tamaño del lote, k recorre los modos almacenados y epsilon es el épsilon de máquina FP32.
Es un cociente espectral de Rayleigh en Å inversos al cuadrado, no una energía de grafo sin
normalizar. Autovalores grandes representan variación superficial rápida. Una constante tiene
energía aproximadamente cero si su modo constante está bien representado. La truncación no ve
frecuencias superiores a las almacenadas; los operadores numéricos pueden dejar residuos pequeños.
El denominador se protege con epsilon cuando la potencia probabilística es cero.

Heat transforma explícitamente **logits durante el forward** mediante exp(−r² Lambda), y el pooling
ve el mapa filtrado. TV y Dirichlet penalizan **probabilidades durante entrenamiento**, sin añadir
por sí mismos un filtro final. En f/g compara, por pooling y semilla: none/lambda0, Heat6/lambda0,
none/pérdida activa, Heat6/pérdida activa. Esos cuatro estados distinguen sustitución, redundancia,
complementariedad y exceso de suavizado. S crudo, S operacional y su diferencia sin recortar muestran
si mejora la evidencia anterior al refinador o solo el mapa filtrado.

Sean L_BCE la BCE global, L_gate la penalización existente de gates semánticos y F las siete familias.
La composición exacta del entrenamiento es:

```math
L=L_{BCE}+0.04L_{gate}+\sum_{f\in F}\lambda_f L_f.
```

En a–g solo varía una familia lambda; las demás quedan a cero. TV y Dirichlet nunca se activan
simultáneamente. Evalúa G, S operacional/crudo, masa/pico negativos, recall/enriquecimiento top-k,
regret de selección, relación global-superficial, dispersión emparejada y coste. Las métricas
indefinidas conservan soportes en lugar de convertirse en cero. Un W alto no justifica por sí solo
añadir complejidad.

## 4. Mezclas exploratorias y revisión de checkpoints

V6h es ahora un **estudio exploratorio de combinaciones**, no una plantilla vacía ni un modelo
final confirmado. Declara 18 candidatos exactos por pooling (54 configuraciones, 216 Runs), todos
con Heat de 6 Å, longitud regional auxiliar de 2.5 Å, margen de ranking 0.5 y el backbone V5c intacto.
`sweep.candidates` ejecuta solo los puntos enumerados: los dominios de parámetros no crean un
producto cartesiano enorme. Los anclajes YAML reutilizan controles a cero sin añadir parámetros científicos.

Los coeficientes siguientes se fijaron tras leer los resúmenes completos V6a–V6f y el seguimiento
experimental de la memoria del TFM el 16 de octubre de 2026. Son hipótesis iniciales, no pesos
optimizados conjuntamente. El informe todavía no contenía resultados de V6g ni de los rangos ampliados.

| Pooling | Negativos N | Existencia puntual E | Existencia regional P | Ranking R | Masa mínima C | TV T | Dirichlet D |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAX | 0.3 | 0.03 | 0.3 | 0.1 | 0.03 | 0.005 | 0.001 |
| LSE por área, beta 0.25 | 1 | 0.03 | 0.1 | 0.03 | 0.1 | 0.005 | 0.001 |
| Attention por área, anchura 256 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.005 | 0.001 |

Cada C activo usa un mínimo de masa probabilística esperada de 0.02, sin máximo. No es un área
de interfaz medida. Los coeficientes multiplican pérdidas con escalas distintas; el mismo valor
numérico no implica la misma intensidad de gradiente.

Los coeficientes difieren entre poolings: compara cada mezcla con su control del mismo pooling.
Una diferencia entre MAX activo y LSE activo no aísla solo el mecanismo de pooling, porque pueden
cambiar también los pesos de las pérdidas. Los controles sin priors conservan esa comparación limpia.

Cada pooling tiene su control cero y los siete términos aislados. El núcleo N/E/R contiene todos
los estados activado/desactivado: control, N, E, R, N+E, N+R, E+R y N+E+R. Otros candidatos
sustituyen E por P, añaden C, T o D por separado al núcleo y acumulan N+E+P+R+C con T o con D.
Estas dos últimas mezclas son pruebas de estrés con todas las familias compatibles, no modelos
recomendados para producción. Nunca se activan TV y Dirichlet juntos. La sustitución E por P ayuda
a identificar redundancia entre dos restricciones de existencia, sin atribuir su suma a un mecanismo nuevo.

Sean A y B dos priors activados con sus coeficientes fijos. Para una puntuación Q, con el mismo
pooling y semilla, se compara la pareja mediante
`interacción = Q(A+B) - Q(A) - Q(B) + Q(control)`. Un valor positivo indica una mejora mayor que
la suma de los cambios individuales en la escala de esa puntuación; uno negativo indica menos.
Si G y S cambian en sentidos opuestos hay un compromiso, no una sinergia universal.
Los controles aislados emparejados deben proceder de V6h, no agruparse con Runs históricos cuyo
código o trayectoria de optimización puedan diferir.

La evidencia es desigual: existencia MAX 0.03 y ranking MAX 0.1 mejoraron métricas locales;
negativos LSE 1 mejoró discriminación global y redujo masa negativa; TV LSE/Heat 0.005 mejoró S
operacional sin mejorar S crudo medio ni limpiar negativos. Atención conserva negativos 0.03
porque los valores observados mayores perjudicaron la discriminación global. Los demás coeficientes
son pruebas conservadoras de sensibilidad, no ganadores acreditados. En particular, C fue a menudo
inactivo en V6e y D=0.001 es provisional hasta revisar V6g. Comprueba las contribuciones registradas
antes de atribuir una mejora a cardinalidad: una contribución cero no demuestra un mecanismo efectivo.
La implementación actual prohíbe combinar las dos penalizaciones de suavidad.

El checkpoint sigue seleccionándose solo por G; test permanece sellado. Examina S operacional/crudo,
G, AUPRC/AUROC globales independientes del umbral, masa/pico negativos, enriquecimiento, regret y
trayectorias temporales, no solo W o accuracy a 0.5. Las semillas históricas 4, 7, 32 y 54 sirven
para comparaciones emparejadas de desarrollo, no para confirmación independiente. Adoptar una mezcla
requiere semillas nuevas y una confirmación congelada aparte. Este YAML no lanza esa confirmación
ni da por disponible la evidencia de V6g.

Los YAML de revisión evalúan checkpoints guardados sin entrenar. Seleccionan una semilla registrada
(los ejemplos usan 4), proteínas contrastantes automáticamente y filtros opcionales de parámetros;
las métricas conservan toda la partición y todos los puntos. Activa canales aprendidos con
`content: features`. Los casos automáticos pueden diferir entre configuraciones: la galería compara
mapas disponibles, no garantiza un panel completo emparejado. Los límites visuales de puntos/bytes
son independientes de la evaluación científica; si faltan pesos se necesita recuperación nativa,
no reentrenamiento oculto. La sección 5.10 del README principal explica la sintaxis actual y las
limitaciones de archivos gestionados en los informes LF.
