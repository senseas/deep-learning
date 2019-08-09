package com.deep.gradient.operation.base;

import lombok.Data;

@Data
public class None {

    public None(Double value) {
        this.value = value;
    }

    private Double value;
    private Double grad = 1d;
}
