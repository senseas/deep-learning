package com.deep.framework.ast.literal;

import com.deep.framework.ast.expression.Expression;

import java.util.regex.Pattern;

public class Literal extends Expression {
    public static final String string = "\"";
    public static final String strings = "\\\"";

    private final String value;

    public Literal(String value) {
        super(null);
        this.value = value;
    }

    public static Literal getLiteral(String value) {
        if (value.equals("null")) {
            return new NullLiteral(value);
        } else if (isBoolean(value)) {
            return new BooleanLiteral(value);
        } else if (isNumber(value)) {
            return new NumberLiteral(value);
        } else if (isString(value)) {
            return new NumberLiteral(value);
        } else if (isChar(value)) {
            return new CharLiteral(value);
        } else {
            throw new IllegalArgumentException("Literal error ".concat(value));
        }
    }

    public static boolean isLiteral(String value) {
        if (isString(value)) return true;
        if (isNumber(value)) return true;
        if (isBoolean(value)) return true;
        if (isNull(value)) return true;
        if (isChar(value)) return true;
        return false;
    }

    public static boolean isNumber(String value) {
        Pattern pattern = Pattern.compile("-?[0-9]+\\.?[0-9]*$");
        return pattern.matcher(value).matches();
    }

    public static boolean isBoolean(String value) {
        return (value.equals("true") || value.equals("false"));
    }

    public static boolean isNull(String value) {
        return value.equals("null");
    }

    public static boolean isChar(String value) {
        return (value.startsWith("'") && value.endsWith("'"));
    }

    public static boolean isString(String value) {
        return (value.startsWith(string) && value.endsWith(string));
    }

    public String getValue() {
        return value;
    }

    public String toString() {
        return value;
    }

}