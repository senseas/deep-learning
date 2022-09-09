package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.RBRACE;

@Data
public class BlockStatement implements Statement {
    public List<Object> statements = new ArrayList<>();

    public void parser(Statement parent, Object obj, List<Object> list) {
        Parser.ends.add(RBRACE);
        parent.statements.add(this);
        while (true) {
            Object o = list.remove(0);
            statements.add(o);
            if (o.equals(Parser.ends.get(0))) {
                return;
            }
        }
    }
}
