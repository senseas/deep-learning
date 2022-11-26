package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;

import static com.deep.framework.ast.lexer.TokenType.*;

public class PrimitiveType extends Type {

    private Type type;

    public PrimitiveType(Name name) {
        super(name);
    }

    /*public static Type getPrimitiveType(Node node) {
        TokenType type = node.getTokenType();
        if (node.equals(INT)) {
            return new IntPrimitiveType(type);
        } else if (node.equals(LONG)) {
            return new LongPrimitiveType(type);
        } else if (node.equals(FLOAT)) {
            return new FloatPrimitiveType(type);
        } else if (node.equals(DOUBLE)) {
            return new DoublePrimitiveType(type);
        } else if (node.equals(BYTE)) {
            return new BooleanPrimitiveType(type);
        } else if (node.equals(CHAR)) {
            return new BytePrimitiveType(type);
        } else if (node.equals(BOOLEAN)) {
            return new BooleanPrimitiveType(type);
        } else {
            return null;
        }
    }*/


    public static boolean isPrimitive(Node node) {
        if (node.equals(INT)) {
            return true;
        } else if (node.equals(LONG)) {
            return true;
        } else if (node.equals(FLOAT)) {
            return true;
        } else if (node.equals(DOUBLE)) {
            return true;
        } else if (node.equals(BYTE)) {
            return true;
        } else if (node.equals(CHAR)) {
            return true;
        } else if (node.equals(BOOLEAN)) {
            return true;
        } else {
            return false;
        }
    }

}
