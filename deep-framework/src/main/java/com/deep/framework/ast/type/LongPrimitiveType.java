package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class LongPrimitiveType extends PrimitiveType {

    public LongPrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}