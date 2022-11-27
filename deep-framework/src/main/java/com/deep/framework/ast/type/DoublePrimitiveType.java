package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class DoublePrimitiveType extends PrimitiveType {

    public DoublePrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}