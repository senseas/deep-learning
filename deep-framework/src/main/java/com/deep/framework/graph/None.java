package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Data
public class None implements Serializable {

    public None(Tensor tensor) {
        this.tensor = tensor;
    }

    public None(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.idx = idx;
    }

    public None(double value) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public None(double value, boolean gradre) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public double getValue() {
        if (Objects.isNull(tensor)) {
            return this.value;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getValue();
        } else {
            return ((double[]) tensor.getValue())[idx];
        }
    }

    public void setValue(double value) {
        if (Objects.isNull(tensor)) {
            this.value = value;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setValue(value);
        } else {
            ((double[]) tensor.getValue())[idx] = value;
        }
    }

    public double getGrad() {
        if (Objects.isNull(tensor)) {
            return this.grad;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getGrad();
        } else {
            return ((double[]) tensor.getGrad())[idx];
        }
    }

    public void setGrad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad += grad;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setGrad((double) tensor.getGrad() + grad);
        } else {
            ((double[]) tensor.getGrad())[idx] += grad;
        }
    }

    public boolean isReduce() {
        if (Objects.isNull(tensor)) {
            return this.reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            return (boolean) tensor.getReduce();
        } else {
            return ((boolean[]) tensor.getReduce())[idx];
        }
    }

    public void setReduce(boolean reduce) {
        if (Objects.isNull(tensor)) {
            this.reduce = reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReduce(reduce);
        } else {
            ((boolean[]) tensor.getReduce())[idx] = reduce;
        }
    }

    public void reset() {
        if (Objects.isNull(tensor)) {
            this.reduce = false;
            this.grad = 0d;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReduce(false);
            tensor.setGrad(0d);
        } else {
            ((boolean[]) tensor.getReduce())[idx] = false;
            ((double[]) tensor.getGrad())[idx] = 0d;
        }
    }

    public void setFuncs(Object... arr) {
        funcx.add(this);
        String code = func;
        func = "";
        for (Object o : arr) {
            if (o instanceof String a) {
                func = func.concat(a);
            } else if (o instanceof None a) {
                if (a.getTensor() instanceof TensorConst) {
                    func = func + a.getValue();
                } else if (a.getFuncx().isEmpty()) {
                    funcx.add(a);
                    func = func.concat("a" + a.getId());
                    param =param.concat(a.getParam());
                } else {
                    funcx.addAll(a.getFuncx());
                    func = func.concat("a" + a.getId());
                    param =param.concat(a.getParam());
                    code = code.concat(a.getFunc());
                }
            }
        }
        param = param.concat("a" + id+",");
        func = "double ".concat("a" + id).concat("=").concat(func);
        func = code.concat("\n").concat(func).concat(";");
        //System.out.println(func);
    }

    public void setGrads(Object... arr) {
        gradx.add(this);
        String code = gradc;
        gradc = "";
        for (Object o : arr) {
            if (o instanceof String a) {
                gradc = gradc.concat(a);
            } else if (o instanceof None a) {
                if (a.getTensor() instanceof TensorConst) {
                    gradc = gradc + a.getValue();
                } else if (a instanceof NoneGrad) {
                    if (a.getGradx().isEmpty()) {
                        gradx.add(a);
                        gradc = gradc.concat("e" + a.getId());
                    } else {
                        gradx.addAll(a.getGradx());
                        gradc = gradc.concat("e" + a.getId());
                        code = code.concat(a.getGradc());
                    }
                } else {
                    gradx.add(a);
                    gradc = gradc.concat("a" + a.getId());
                }
            }
        }
        gradc = "double ".concat("e" + id).concat("=").concat(gradc);
        gradc = code.concat("\n").concat(gradc).concat(";");
        System.out.println(gradc);

        if(!gradx.isEmpty()){
            None none = Stream.of(gradx).flatMap(Collection::stream).filter(a -> a instanceof NoneGrad).findFirst().get();
            String a =
            "class Tensor {\n" +
            "  double " + none.getParam() + ";\n" +
            "  void compute(double a"+ id +") {\n" +
            "    " + none.getFunc() +
            "  }\n" +
            "  void gradient(double a"+ id +", double e"+none.getId()+") {\n" +
            "    " +gradc +
            "  }\n" +
            "};\n";
            System.out.println(a);
        }
    }

    public None grad() {
        return new NoneGrad(this);
    }

    private int idx;
    private transient Tensor tensor;
    private double value, grad;
    private transient boolean reduce;
    private int id = ID.getAndIncrement();
    private String param = "";
    private transient String func = "", gradc = "";
    private transient List<None> funcx = new ArrayList<>(), gradx = new ArrayList<>();
    public static AtomicInteger ID = new AtomicInteger();
}
