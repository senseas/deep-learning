package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class BooleanPrimitiveType extends PrimitiveType {

    public BooleanPrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}
