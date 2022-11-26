package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class IntPrimitiveType extends PrimitiveType {

    public IntPrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}
