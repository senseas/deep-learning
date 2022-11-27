package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class FloatPrimitiveType extends PrimitiveType {

    public FloatPrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}