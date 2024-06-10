package com.deep.framework.lang.util;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

public class Streams {

    private static final int threads = Runtime.getRuntime().availableProcessors();
    private static final ExecutorService executor = Executors.newFixedThreadPool(threads);

    public static void forEach(int workNum, IntConsumer action) {
        List<Future> list = new ArrayList<>();
        for (int i = 0; i < workNum; i++) list.add(executor.submit(work(i, action)));
        synchronize(list);
    }

    private static Runnable work(int i, IntConsumer action) {
        return () -> {
            action.accept(i);
        };
    }

    public static <M> void forEach(M[] data, Consumer<M> action) {
        List<Future> list = new ArrayList<>();
        for (M m : data) list.add(executor.submit(work(m, action)));
        synchronize(list);
    }

    private static <M> Runnable work(M m, Consumer<M> action) {
        return () -> {
            action.accept(m);
        };
    }

    private static void synchronize(List<Future> list) {
        for (Future a : list) {
            try {
                a.get();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

}