package com.deep.framework.ast.expression;

public class StringExpression extends Expression {
    private final String value;

    public StringExpression(String value) {
        super(null);
        this.value = value;
    }

    public String toString() {
        return value;
    }
}