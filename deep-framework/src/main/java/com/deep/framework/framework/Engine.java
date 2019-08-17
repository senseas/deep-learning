package com.deep.framework.framework;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;
import com.deep.framework.graph.Graph;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tenser;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;
import org.apache.log4j.Logger;

@Data
public class Engine extends Shape {
    Logger log = Logger.getLogger(Engine.class);
    public static double rate = 0.25;

    public void forward(Tenser tenser) {
        execute(tenser, a -> {
            Tenser<None> node = (Tenser) a;
            computer(node, node.getOutput().getGraph());
            compute(node);
        }, a -> {
            Tenser<None> node = (Tenser) a;
            graph(node, node.getGraph());
            function(node);
        });
        //log.info(JSONObject.toJSONString(tenser));
    }

    private void computer(Tenser tenser, Graph graph) {
        for (Node o : tenser.getInput()) {
            Tenser node = (Tenser) o;
            if (BeanUtil.isNotNone(node)) {
                if (BeanUtil.isNotOperation(node)) {
                    Tenser func = (Tenser) node.getFunction();
                    computer(func, graph);
                    compute(func);
                    graph.add(func);
                } else {
                    computer(node, graph);
                    compute(node);
                    graph.add(node);
                }
            }
        }
    }

    private void compute(Tenser<None> tenser) {
        None nones = tenser.compute(), outputs = tenser.getOutput();
        Func2<None, None> func = (none, out) -> {
            out.setValue(none.getValue());
        };
        forEach(nones, outputs, func);
    }

    private void graph(Tenser tenser, Graph graph) {
        for (Node o : tenser.getInput()) {
            Tenser<Tenser> a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                graph(a, graph);
                function(a);
                graph.add(a);
            }
        }
    }

    private void function(Tenser tenser) {
        Func2<Tenser<None>, None> func = (node, out) -> {
            computer(node, node.getGraph());
            compute(node);
            out.setValue(node.getOutput().getValue());
        };
        forEach(tenser.getFunction(), tenser.getOutput(), func);
    }

    public void backward(Tenser<None> tenser) {
        execute(tenser, a -> {
            gradient(a);
            Func1<None> func = out -> {
                out.getGraph().farEach(o -> {
                    gradient(o);
                });
            };
            forEach(a.getOutput(), func);
        }, a -> {
            gradients(a);
            a.getGraph().farEach(o -> {
                gradients((Tenser) o);
            });
        });
        //log.info(JSONObject.toJSONString(tenser));
        _backward(tenser);
    }

    private void gradient(Tenser<None> tenser) {
        tenser.gradient();
        tenser.getOutput().setGrad(null);
    }

    private void gradients(Tenser tenser) {
        Func2<Tenser<None>, None> func = (node, out) -> {
            node.getOutput().setGrad(out.getGrad());
            gradient(node);
            node.getGraph().farEach(o -> {
                gradient(o);
            });
        };
        forEach(tenser.getFunction(), tenser.getOutput(), func);
    }

    private void _backward(Tenser tenser) {
        execute(tenser, a -> {
            reduce(a);
            Func1<None> func = out -> {
                out.getGraph().farEach(o -> {
                    reduce(o);
                });
            };
            forEach(a.getOutput(), func);
        }, a -> {
            reduceTenser(a);
            a.getGraph().farEach(o -> {
                reduceTenser((Tenser) o);
            });
        });
    }

    private void reduceTenser(Tenser tenser) {
        Func1<Tenser<None>> func = node -> {
            reduce(node);
            node.getGraph().farEach(o -> {
                reduce(o);
            });
        };
        forEach(tenser.getFunction(), func);
    }

    private void reduce(Tenser tenser) {
        Func1<Tenser> func = node -> {
            forEach(node.getOutput(), (Func1<None>) a -> {
                if (BeanUtil.startsWithNone(node)) {
                    Double value = a.getValue() - rate * a.getGrad();
                    a.setValue(value);
                }
                a.setGrad(null);
            });
        };
        forEach(tenser.getInput(), func);
    }

    private void execute(Tenser tenser, Func1<Tenser>... func) {
        if (BeanUtil.isOperation(tenser)) {
            func[0].apply(tenser);
        } else {
            func[1].apply(tenser);
        }
    }

    public void init(Tenser a, Object b) {
        Func2<None, Double> func = (m, n) -> {
            m.setValue(n);
        };
        forEach(a.getOutput(), b, func);
    }
}
