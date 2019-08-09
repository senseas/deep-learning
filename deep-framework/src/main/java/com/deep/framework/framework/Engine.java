package com.deep.framework.framework;

import com.deep.framework.bean.None;
import com.deep.framework.graph.Tenser;
import com.deep.framework.lang.ForEach;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;
import org.apache.log4j.Logger;

@Data
public class Engine extends ForEach {
    Logger log = Logger.getLogger(Engine.class);
    public static double rate = 0.25;

    public void forward(Tenser tenser) {
        execute(tenser, a -> {
            Func1<None> func = out -> {
                out.getGraph().forEach(o -> {
                    compute(o);
                });
            };
            forEach(a.getOutput(), func);
            compute(a);
        }, a -> {
            Func1<Tenser<None>> func = node -> {
                node.getOutput().getGraph().forEach(o -> {
                    compute(o);
                });
                compute(node);
            };
            forEach(a.getFunction(), func);
        });
        //log.info(JSONObject.toJSONString(tenser));
    }

    private void compute(Tenser<None> tenser) {
        None nones = tenser.compute(), outputs = tenser.getOutput();
        Func2<None, None> func = (none, out) -> {
            out.setValue(none.getValue());
        };
        forEach(nones, outputs, func);
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
            Func1<Tenser<None>> func = node -> {
                gradient(node);
                node.getOutput().getGraph().farEach(o -> {
                    gradient(o);
                });
            };
            forEach(a.getFunction(), func);
        });
        //log.info(JSONObject.toJSONString(tenser));
        _backward(tenser);
    }

    private void gradient(Tenser<None> node) {
        node.gradient();
        node.getOutput().setGrad(null);
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
            Func1<Tenser<None>> func = node -> {
                reduce(node);
                node.getOutput().getGraph().farEach(o -> {
                    reduce(o);
                });
            };
            forEach(a.getFunction(), func);
        });
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
