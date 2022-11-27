package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class BytePrimitiveType extends PrimitiveType {

    public BytePrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}