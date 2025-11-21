# Diario de Trabajo
## En este documento los desarolladores del trabajo documentarán el trabajo realizado por cada uno de ellos
### A la hora de usar el "Diario de Trabajo" sugerimos seguir las siguientes indicaciones:
- Añade contenido siempre que vayas a hacer el último commit de tu sesión de trabajo para que todo el mundo sepa que ha sido modificado.
- Usa tu nombre/inicial/acrónimo y fecha de sesion al lado de las tareas que modifiques para que todos sepan quien ha sido el último en trabajar en ellas y pueda preguntar dudas directamente.
- Pese a que las explicaciones no sean malas, siempre es mejor dar detalles sobre el funcionamiento del código en el mismo codigo. Utiliza el diario para datos generales, indicaciones de posibles acciones para el futuro y comentarios sobre los resultados obtenidos.

### Lista de tareas:

#### Tareas empezadas:

#### Tareas Completadas:

#### Tareas Testeadas:

#### Tareas Terminadas

### Diario de Trabajo:

#### Sesión Cristian - 21/11
He estructurado la BD de cara a trabajar con el AE. De cara a entrenamiento se usa única y exclusivamente los pacientes sanos del Cropped. Hay unos 70000 y los dividimos de diferentes maneras en función de si usamos el método de Cross-Validation con K-Folds o un entrenamiento clásico. En el caso de usar K-Folds los pacientes **(importante tener en cuenta que las divisiones se hacen por paciente no solo por imagenes ya que las imagenes de un solo paciente pueden ser bastante similares entre ellas y así nos ahorramos posibles casos de data leackage)** se dividen en K grupos de más o menos el mismo número de imagenes y para cada fold se escoge un grupo que actuará como validation group, mientras que los demás serán el training group. De cara al entrenamiento clásico los pacientes de Cropped se dividen 80-20 entre un train y un test. Después las imagenes del annotated se usarán para ver el comportamiento del AE parche a parche y para decidir métricas como el threshold de error a partir del cual consideramos a un parche como contaminado (esta parte está empezada a implementar pero no acabada porque los caminos del annotated me estaban dando varios problemas). El conjunto de HoldOut se reserva para hacer un último test clínico de la precisión del modelo a la hora de diagnosticar pacientes, NO PARCHES. De cara a las estructuras del codigo se ha modificado el ImageDataset para que ahora no verifique cada imagen que se carga cada vez, ya que al hacer el kfold se verificaba cada imagen del dataset k veces y provocaba un cuello de botella enorme. También he añadido parámetros a los DataLoaders para que utilicen de forma más eficiente los workers y se comuniquen mejor con la GPU para agilizar los procesos de entrenamiento (no obstante hay que ir con cuidado cuando se ejecute en la máquina por las incompatibilidades entre Linux y Windows para estas weas). De cara al entrenamiento he hecho una función que se encarga de entrenar un modelo, así se puede llamar k veces cuando se hace el kfolds o una cuando se hace el entrenamiento clásico y he hecho un entrenamiento kfolds los resultados del cual estan en Discord. Desgraciadamente pensaba que el código sería más corto porque en el portatil las cosas se ven como el culardo pero desde el ordenado he visto que he creado una pequeña abominación de código, así que uno de estos días me encargaré de parametrizarlo con tal de que la estructura y cada función sea más clara. Fuera de esto no hay mucho más. Ya hay algunos errores calculados en imagenes contaminadas dentro de la carpeta Results en AutoEncoders y los csv relevantes que he creado estan en Discord.