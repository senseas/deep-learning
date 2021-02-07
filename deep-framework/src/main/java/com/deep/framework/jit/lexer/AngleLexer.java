package com.deep.framework.jit.lexer;

import lombok.Data;

@Data
public class AngleLexer implements Lexer {
    private String type = "shift";
    private Lexer lexer;
    public transient final static String start = "<", end = ">";

    public AngleLexer(Lexer lexer) {
        this.lexer = lexer;
    }

    public String getEnd() {
        return end;
    }
}