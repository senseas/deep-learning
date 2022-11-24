package com.deep.framework.ast.expression;

public class StringLiteralExpression extends Expression {
    private final String value;

    public StringLiteralExpression(String value) {
        super(null);
        this.value = value;
    }

    public String toString() {
        return value;
    }
}