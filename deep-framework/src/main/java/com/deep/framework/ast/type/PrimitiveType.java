package com.deep.framework.ast.type;

import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;

import static com.deep.framework.ast.lexer.TokenType.*;

public class PrimitiveType extends Type {

    public PrimitiveType(TokenType type) {
        super(new Name(type));
    }

    public static Type getPrimitiveType(TokenType type) {
        if (type.equals(SHORT)) {
            return new ShortPrimitiveType(type);
        } else if (type.equals(INT)) {
            return new IntPrimitiveType(type);
        } else if (type.equals(LONG)) {
            return new LongPrimitiveType(type);
        } else if (type.equals(FLOAT)) {
            return new FloatPrimitiveType(type);
        } else if (type.equals(DOUBLE)) {
            return new DoublePrimitiveType(type);
        } else if (type.equals(BYTE)) {
            return new BooleanPrimitiveType(type);
        } else if (type.equals(CHAR)) {
            return new CharPrimitiveType(type);
        } else if (type.equals(BOOLEAN)) {
            return new BooleanPrimitiveType(type);
        } else {
            return null;
        }
    }
}