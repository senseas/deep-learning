package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class ShortPrimitiveType extends PrimitiveType {

    public ShortPrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}