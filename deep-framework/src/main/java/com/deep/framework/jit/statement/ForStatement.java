package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;

import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.SEMI;

public class ForStatement implements Statement {
    public BlockStatement param = new BlockStatement();
    public BlockStatement block = new BlockStatement();

    public void parser(Statement parent, Object obj, List<Object> list) {
        Parser.ends.add(SEMI);
        Parser.statements.add(this);
        while (true) {
            Object o = list.remove(0);
            if (o.equals(Parser.ends.get(0))) {
                Parser.ends.remove(0);
                return;
            } else {
                block = (BlockStatement) o;
            }
        }
    }
}
