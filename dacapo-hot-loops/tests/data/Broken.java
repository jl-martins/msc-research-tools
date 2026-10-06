package p;

class Broken {
    void loop() {
        for (int i = 0; i < 2; i++) {
            i++;
        }
    }

    void unterminated() {
