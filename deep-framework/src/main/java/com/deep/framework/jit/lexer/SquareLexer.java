package com.deep.framework.jit.lexer;

import lombok.Data;

@Data
public class SquareLexer implements Lexer {
    private String type = "dims";
    private Lexer lexer;
    public transient final static String start = "[", end = "]";

    public SquareLexer(Lexer lexer) {
        this.lexer = lexer;
    }

    public String getEnd() {
        return end;
    }
}