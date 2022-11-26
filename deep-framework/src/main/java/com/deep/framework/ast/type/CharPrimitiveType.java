package com.deep.framework.ast.type;

import com.deep.framework.ast.lexer.TokenType;

public class CharPrimitiveType extends PrimitiveType {

    public CharPrimitiveType(TokenType type) {
        super(type);
        setTokenType(type);
    }

}
