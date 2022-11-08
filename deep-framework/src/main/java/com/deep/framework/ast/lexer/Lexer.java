package com.deep.framework.ast.lexer;

public interface Lexer {

    String getType();

    String getEnd();

    Lexer getLexer();

}