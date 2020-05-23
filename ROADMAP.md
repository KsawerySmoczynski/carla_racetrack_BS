# Plan na najbliższy czas
     * skrypt do oceny sumarycznej nagrody z episodu każdego aktora, gdy średnia nagroda jest niższa niż 0.01 to usuwamy pliki takiego episodu, że nie brało ich jako buffer 
    
## Uczymy sieć na mpc
    - Przeczytaj train_on_depth (https://github.com/asap-report/carla/tree/racetrack/PythonClient/racetrack)! -> jego sieć prognozuje trasę, to nie jest reinforcement, 
    dowiedz się jak przetworzyć ciągłe wartości na wartości akcji w dqn. -> papiery które Jacek przeczytał. 
    - Trenujemy na 4 torach i testujemy na 2
    - Okręślamy hiperparametry
    - Zapisujemy najlepszy model
    - spójrz na mechanizm weights -> gradient się poprostu bardziej liczy

# Done 
## Stworzenie MPC jeżdżącego po torze
- [Najlepsze](https://towardsdatascience.com/the-final-step-control-783467095138)
- [YT](https://www.youtube.com/watch?v=nqv6jFeVUYA)
- [YT cars](https://www.youtube.com/watch?v=Gh8R4PVg1Zc)
- [Model Predictive Controller Project](https://medium.com/@NickHortovanyi/carnd-controls-mpc-2f456ce658f)
- [Up](https://github.com/hortovanyi/CarND-MPC-Project)
- [Model Predictive Control for Autonomous Vehicles](https://medium.com/@shubhra.pandit/model-predictive-control-for-autonomous-vehicles-1dc18348f651)
     
* Utworzyć system logowania informacji i model danych
* Tensorboard na zmienne parametry -> prędkość, wartości kontrolerów ogarnąć jak działa GlobalSummaryWriter
* Visdom -> do wizualizacji obrazów i lokalizacji auta na torze. Nie powielać informacji, wysyłać zapisane z dysku na front. Jak się nie uda to przemyśleć jak uprościć model
* Puścić MPC model w każdej konfiguracji i zapisać wszystkie informacje 
* Napisać skrypt do visdom który na podstawie danych spływających serwer prowadziłby logowanie.

# Inne

- Jak zaprojektować funkcję celu?
Potencjalne problemy z f. celu:
    1. Agent nie uczy się, że warto wciskać gaz bo nie widzi różnicy z wrzuceniem luzu
    2. Agent uczy się przyspieszać i zwalniać, aby nadużywać składnika nagrody będącego iloczynem prędkości poprzedniej i aktualnej zamiast jedynie przyspieszać
    3. Coś było z zakrętami

- Jak wpiąć MPC aby była mierzalna za pomocą funkcji celu.
- Które algorytmy obsługują continous action space.


(Odległość do najbliższego punktu + suma odległości pozostałych punktów) / czas symulacji
- wymaga dostarczenia informacji autu o odległości do najbliższego punktu / do końca, oraz azymutu do kolejnego? mniej punktów


- Stan w jakim jest agent musi obejmować prędkość skręt i odległość do wyliczenia f-celu -> może też to być po prostu startowy spawn point
- Samplując z przestrzeni stanów (replay buffer) wartość otrzymamy za pomocą miary odległości od 'końca'.
- Czy podajemy też azymut? nie wydaje się to konieczne gdy sieć będzie uczyła się na bufferze z MPC

- wykonując krok w środowisku:
    * ustawiamy gaz i skręt aktora i robimy world tick
    * wyliczamy odległość do "mety" przez wszystkie punkty -> punktów musi być na tyle dużo, że każde posunięcie w przód powoduje określenie najbliższego punktu jako kolejnego na trasie
    * ewentualnie można liczyć czy aktor "przekroczył" najbliższy punkt -> patrząc czy jest bliższa odległość do kolejnego czy do poprzedniego. (Punkty muszą być rozłożone w równych odległościach od siebie).
    
- Loss liczymy pomiędzy wartością wynikającą z modelu -> początkowo MPC, a potem z sieci.
- Jesteśmy w stanie równolegle zrzucać auta w losowych 50 miejscach i liczyć loss z każdego z działań. przeprowadzać cały batch jednocześnie.

# READ
- [Thesis](https://esc.fnwi.uva.nl/thesis/centraal/files/f285129090.pdf)