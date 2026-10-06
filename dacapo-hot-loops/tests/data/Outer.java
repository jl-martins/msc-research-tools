package p;

class Outer {
    static int total = init();

    static {
        total++;
    }

    Outer() {
        for (int i = 0; i < 2; i++) total++;
    }

    void work() {
        for (int i = 0; i < 3; i++) {
            while (total > 0) {
                total--;
            }
        }
        do {
            total++;
        } while (total < 5);
        Runnable lambda = () -> {
            for (String name : names) {
                use(name);
            }
        };
        new Thread() {
            public void run() {
                for (int k = 0; k < 4; k++) {
                    total--;
                }
            }
        };
    }

    void work(int times) {
        total += times;
    }

    class Inner {
        void spin() {
            while (true) {
                total++;
            }
        }
    }
}
