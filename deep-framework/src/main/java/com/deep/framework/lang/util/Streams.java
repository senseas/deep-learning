package com.deep.framework.lang.util;

import com.deep.framework.graph.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.stream.Stream;

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

    public static void forEach(Tensor[] data, Consumer<Tensor> action) {
        List<Future> list = new ArrayList<>();
        List<Tensor> tensors = new CopyOnWriteArrayList<>(data);
        while (!tensors.isEmpty()) {
            for (Tensor o : tensors) {
                if (Stream.of(o.getInput()).parallel().filter(a -> Objects.nonNull(a.getInput()) || Objects.nonNull(a.getTensor())).anyMatch(a -> !a.isStatus())) continue;
                if (Objects.nonNull(o.getFunction()) && o.getFunction().stream().parallel().anyMatch(a -> !a.isStatus())) continue;

                list.add(executor.submit(work(o, action)));
                tensors.remove(o);
            }
        }
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