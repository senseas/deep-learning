package com.deep.framework.ast.lexer;

public class Token{

    private TokenType tokenType;

    public Token(TokenType type) {
        this.tokenType = type;
    }

    public TokenType getTokenType() {
        return tokenType;
    }

    public void setTokenType(TokenType tokenType) {
        this.tokenType = tokenType;
    }

    @Override
    public String toString() {
        return tokenType.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return tokenType == o;
    }
}