package com.deep.framework.graph;

import com.deep.framework.ast.CompilationUnit;
import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import java.lang.*;

public class TensorOparetor extends Tensor {

    public enum Color {
        RED("RED") {
            @java.lang.Override()
            public String toString() {
                return super.toString();
            }
        }, BLUE("BLUE"), GREEN("GREEN"), BLACK("BLACK");

        Color(String name) {
            this.name = name;
        }

        private String name;
    }

    public enum Color2 {
        RED, BLUE, GREEN, BLACK;

        private String name;
    }

    Double aa = -100.2;
    Boolean bb = false;
    char cc = '1';
    public String x = "xxxxx";
    public int b = cc == 2 ? 1 : 2 + 1 * 7;

    public TensorOparetor(String name, Tensor... input) {
        super(name, input);
        System.out.println(CompilationUnit.arr[1]);
        Node node = new Node() {
            public NodeList split(Node node) {
                return super.split(node);
            }
        };
        switch (name) {
            case "1":
                break;
            case "2":
                System.out.println(1);
                break;
            case "3":
                System.out.println(1);
                break;
            default:
                System.out.println(1);
                break;
        }
        try {
            System.out.println(1);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            System.out.println(1);
        }
        synchronized (this) {
            System.out.println(1);
        }

        do {
            int a = +b;
            System.out.println("do while");
        } while (1 == 2);

        List<List<java.lang.Object>> list;
        if (1 > 1) {
            System.out.println(1);
        } else if (1 > 2) {
            System.out.println(2);
        } else {
            System.out.println(3);
        }

        if (Arrays.asList("Add").contains(name)) {
            Stream<Tensor> stream = Stream.of();
            for (Tensor o : input) {
                Stream<Tensor> children = o.getName().equals(getName()) ? Arrays.stream(o.getInput()) : Stream.of(o);
                stream = Stream.concat(stream, children);
            }
            Stream.of(1, 1, 1).map((Integer a) -> {
                return a;
            });
            Stream.of(1, 1, 1).map(a -> a);
            setInput(stream.toArray(Tensor[]::new));
            final int a = 1 + 2 * 3 * 4 + 5 + 7 * 8;
        }

        for (int i = 0; i < 100; i++) {
            System.out.println("int i = 0; i < 100; i++");
        }

        while (aa == 11) {
            System.out.println("while");
        }
        int[] arr = new int[2];
        int[][][] arx = new int[2][1][22];
        int a = 1;
        System.out.println(arr[a + 0]);

        throw new RuntimeException("1");

    }

    public None getInput(int i) {
        Tensor input = getInput()[i];
        return (None) input.getOutput();
    }

    public None getOutput() {
        return (None) output;
    }

    public void forward() {
        for (Tensor o : getInput()) TensorFlux.computer(o);
        TensorFlux.compute(this);
    }

    public void backward() {
        TensorFlux.gradient(this);
        for (Tensor o : getInput()) o.backward();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        for (Tensor o : getInput()) TensorFlux.reducer(o);
    }

}