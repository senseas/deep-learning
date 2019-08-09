package com.deep.gradient.operation.base;

import lombok.Data;

@Data
public class None_OP {

    public None_OP(Double value) {
        this.value = value;
    }

    private Double value;
    private Double grad = 1d;
}
