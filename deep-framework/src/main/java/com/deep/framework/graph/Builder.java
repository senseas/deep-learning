package com.deep.framework.graph;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.util.BeanUtil;

import java.util.stream.Stream;

public class Builder extends Shape {

    public static void create(Tenser tenser) {
        if (BeanUtil.isNotOperation(tenser)) {
            Object function = tenser.compute();
            if (BeanUtil.isNotTenser(function)) {
                Tenser tense = (Tenser) function;
                if (BeanUtil.isNotOperation(tense)) {
                    tenser.setFunction(tense.getFunction());
                } else {
                    tenser.setFunction(function);
                }
            } else {
                tenser.setFunction(functions(function));
            }
        } else {
            operator(tenser);
        }
    }

    private static void operator(Tenser tenser) {
        None none = new None(0d);
        Graph graph = none.getGraph();
        tenser.setOutput(none);
        Func1<Tenser> func = node -> {
            if (BeanUtil.isNotNone(node)) {
                None out = (None) node.getOutput();
                graph.addAll(out.getGraph());
                if (BeanUtil.isOperation(node)) {
                    graph.add(node);
                } else {
                    graph.add(node.getFunction());
                }
            }
            if (BeanUtil.isAdd(tenser) && BeanUtil.isAdd(node)) {
                Stream<Node> a = Stream.of(node.getInput());
                Stream<Node> b = Stream.of(tenser.getInput()).filter(o -> o != node);
                tenser.setInput(Stream.concat(a, b).toArray(Tenser[]::new));
                graph.remove(node);
            }
        };
        forEach(tenser.getInput(), func);
    }

    public static <M> M build(Tenser tenser, int i) {
        Tenser input = (Tenser) tenser.getInput()[i];
        if (BeanUtil.isOperation(tenser)) return (M) input.getOutput();
        if (BeanUtil.isNone(input)) return Shape.tensers(input.getOutput());
        if (BeanUtil.isOperation(input)) return (M) input;
        return (M) input.getFunction();
    }
}
