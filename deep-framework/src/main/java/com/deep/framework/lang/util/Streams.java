package com.deep.framework.lang.util;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

public class Streams {

    private static int threads = Runtime.getRuntime().availableProcessors();
    private static final ExecutorService executor = Executors.newFixedThreadPool(threads);
    private final int workNum;
    private final AtomicInteger count;

    public Streams(int workNum) {
        this.workNum = workNum;
        this.count = new AtomicInteger(0);
    }

    public void forEach(IntConsumer action) {
        for (int i = 0; i < workNum; i++) {
            executor.submit(work(i, action));
        }
        synchronize();
    }

    private Runnable work(int i, IntConsumer action) {
        return () -> {
            action.accept(i);
            count.getAndIncrement();
        };
    }

    private void synchronize() {
        while (true) {
            if (count.get() >= workNum) break;
        }
    }

}