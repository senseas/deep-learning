package com.deep.framework.ast.expression;

public class StringExpression extends Expression {
    private String value;

    public StringExpression(String value) {
        super(null);
        this.value = value;
    }

    public String toString() {
        return value;
    }
}