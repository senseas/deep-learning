package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import lombok.Data;

import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.CASE;
import static com.deep.framework.ast.lexer.TokenType.DEFAULT;

@Data
public class SwitchEntry extends Statement {

    private static SwitchEntry statement;

    public static void parser(Node node) {
        statement = null;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(CASE)) {
                    //create ForNode and set Prarent，Parameters
                    statement = new SwitchEntry();
                    statement.setPrarent(node);
                    node.replace(a, statement);

                    //remove ForNode and Parameters
                    a.getChildrens().remove(m);
                } else if (m.equals(DEFAULT)) {
                    //create ForNode and set Prarent，Parameters
                    statement = new SwitchEntry();
                    statement.setPrarent(node);
                    node.replace(a, statement);

                    //remove ForNode and Parameters
                    a.getChildrens().remove(m);
                } else if (Objects.nonNull(statement)) {
                    statement.getChildrens().add(m);
                    a.getChildrens().remove(m);
                    if (a.getChildrens().isEmpty()) node.getChildrens().remove(a);
                }
            });
        });
    }

}