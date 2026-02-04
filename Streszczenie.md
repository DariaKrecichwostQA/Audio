# Jak działa model AudioSentinel (Wersja LSTM)?

AudioSentinel wykorzystuje sieć rekurencyjną typu **LSTM (Long Short-Term Memory)** do wykrywania nieprawidłowości w pracy maszyn. W przeciwieństwie do prostych modeli, ta wersja rozumie **kontekst czasowy** dźwięku.

## 1. Architektura: LSTM Autoencoder
Model nie analizuje pojedynczych chwil, lecz **sekwencje ramek** (okna czasowe).
*   **Pamięć modelu:** Warstwy LSTM posiadają wewnętrzne "komórki pamięci", które pozwalają im śledzić trendy w sygnale (np. narastające tarcie czy cykliczność pracy silnika).
*   **Proces:** Wejście (sekwencja 10 ramek) jest kompresowane do wektora kontekstu, a następnie model próbuje "przewidzieć" i odtworzyć całą tę sekwencję.

## 2. Co to daje w praktyce?
Zastosowanie sieci rekurencyjnej pozwala wykryć anomalie, które są "poprawne" w danej chwili, ale "błędne" w kontekście czasu:
*   **Zmiany rytmu:** Maszyna pracuje za szybko lub za wolno, mimo że każdy pojedynczy dźwięk brzmi normalnie.
*   **Nieregularne stukanie:** Model zauważa brak przewidywalności w sygnale.
*   **Ewolucja usterki:** Model lepiej wychwytuje powolne zmiany w barwie dźwięku (np. przegrzewające się łożysko).

## 3. Detekcja Anomalii Czasowych
System wylicza **Błąd Rekonstrukcji Sekwencji**. Jeśli model "zapomni" wzorca rytmicznego maszyny, różnica między wejściem a wyjściem gwałtownie rośnie, co aktywuje alarm.

## 4. Diagnostyka Gemini AI
Gemini AI otrzymuje teraz bardziej precyzyjne dane o **dynamice zdarzeń**. Może interpretować, czy anomalia jest nagła (udarowa), czy ma charakter narastający, co pozwala na znacznie dokładniejszą diagnozę usterek mechanicznych.
