package com.deep.framework.jit.statement;

import com.deep.framework.jit.lexer.TokenType;
import lombok.Data;

import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.CLASS;
import static com.deep.framework.jit.lexer.TokenType.RBRACE;

@Data
public class ClassStatement implements Statement {
    public List<FunctionStatement> implement;
    public List<FunctionStatement> extend;
    public BlockStatement block;

    public String access;
    public String name;
    public TokenType start = CLASS, end = RBRACE;

    public void parser(Statement parent, Object obj, List<Object> lexers) {
        if (obj.equals(CLASS)) {
            parent.statements.add(this);
            StringBuilder buffer = new StringBuilder();
            while (true) {
                Object o = lexers.get(0);
                lexers.remove(0);
               /* if (o.equals(BlockStatement.tokenType)) {
                    name = buffer.toString();
                    block = new BlockStatement();
                    block.parser(parent, o, lexers);
                    return;
                } else */
                if (o.equals(RBRACE)) {
                    name = buffer.toString();
                    block = new BlockStatement();
                    block.parser(parent, o, lexers);
                    return;
                } else {
                    buffer.append(o);
                }
            }
        }
    }
}
