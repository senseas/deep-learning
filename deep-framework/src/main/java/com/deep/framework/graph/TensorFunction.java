package com.deep.framework.graph;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.lang.util.BeanUtil;
import org.apache.log4j.Logger;

import java.util.Objects;
public class TensorFunction extends Tensor {
    static Logger log = Logger.getLogger(TensorFunction.class);
    public TensorFunction(String name, Tensor... input) {
        super(name, input);
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        if (BeanUtil.isOperation(input)) return (M) input;
        if (BeanUtil.isFunction(input)) return (M) input.getFunction();
        return (M) Shape.tensors(input.getOutput());
    }

    public Object getOutput() {
        if (Objects.nonNull(output)) return output;
        if (Objects.nonNull(getFunction())) output = Builder.getOutput(function);
        return output;
    }

    public Object getFunction() {
        if (Objects.nonNull(function)) return function;
        Object functions = compute();
        if (Objects.nonNull(functions)) function = Shape.functions(functions);
        return function;
    }

    public void computeing() {
        Shape.farEach(getInput(), o -> ((Tensor) o).computeing());
        Builder.function(this);
    }

    public void gradienting() {
        Builder.gradientFunction(this);
        Shape.farEach(getInput(), o -> ((Tensor) o).gradienting());
    }

    public void reduceing() {
        Builder.reducerFunction(this);
        Shape.farEach(getInput(), o -> ((Tensor) o).reduceing());
    }

}
