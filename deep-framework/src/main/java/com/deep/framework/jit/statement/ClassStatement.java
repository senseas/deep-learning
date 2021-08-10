package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import lombok.Data;

import java.util.List;

import static com.deep.framework.jit.lexer.TokenType.CLASS;

@Data
public class ClassStatement implements Statement {
    public List<FunctionStatement> implement;
    public List<FunctionStatement> extend;
    public BlockStatement block;

    public String access;
    public String name;

    public void parser(Statement parent, Object obj, List<Object> list) {
        if (obj.equals(CLASS)) {
            parent.statements.add(this);
            StringBuilder buffer = new StringBuilder();
            while (true) {
                Object o = list.get(0);
                list.remove(0);
                if (o instanceof BlockStatement) {
                    block = (BlockStatement) o;
                    return;
                } else {
                    buffer.append(o);
                }
            }
        }
    }
}
