package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.List;

@Data
public class ImportStatement implements Statement {
    public String name;

    public void parser(Statement parent,Object obj, List<Object> list) {
        if (obj.equals(TokenType.IMPORT)) {
            Parser.statementList.add(this);
            StringBuilder buffer = new StringBuilder();
            while (!list.isEmpty()) {
                Object o = list.get(0);
                list.remove(0);
                if (o.equals(TokenType.SEMI)) {
                    name = buffer.toString();
                    return;
                } else {
                    buffer.append(o);
                }
            }
        }
    }

}
