package com.deep.framework.jit.lexer;

public interface Lexer {

    String getType();

    String getEnd();

    Lexer getLexer();

}